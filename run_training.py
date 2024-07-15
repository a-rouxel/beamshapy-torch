import matplotlib.pyplot as plt
from units import *
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from helpers import load_yaml_config
from simulation import Simulation
from sources import Source
from slm import SLM
from ft_lens import FT_Lens
from FieldGeneration import generate_target_profiles
from torch.utils.tensorboard import SummaryWriter
from cost_functions import *



class FixedDataset(Dataset):
    def __init__(self, source_field, list_target_field):
        self.source_field = source_field
        self.list_target_field = list_target_field

    def __len__(self):
        return 1  # Only one element in the dataset

    def __getitem__(self, idx):
        return self.source_field, self.list_target_field

class OpticalSystem(pl.LightningModule):
    def __init__(self,log_dir="./logs",device="cpu",target_mode_nb=2):
        super().__init__()
        self.config_source = load_yaml_config("./configs/source.yml")
        self.config_slm = load_yaml_config("./configs/SLM.yml")
        self.config_simulation = load_yaml_config("./configs/simulation.yml")

        self.simulation = Simulation(config_dict=self.config_simulation)
        self.source = Source(config_dict=self.config_source, XY_grid=self.simulation.XY_grid)
        self.ft_lens = FT_Lens(self.simulation.delta_x_in, self.simulation.XY_grid, self.source.wavelength)
        # Freeze the seg_model parameters
        # for param in self.source.parameters():
        #      param.requires_grad = False

        self.target_mode = target_mode_nb
        self.list_target_files = generate_target_profiles(yaml_file="./configs/target_profile.yml",
                                                     XY_grid=self.ft_lens.XY_output_grid,
                                                     list_modes_nb=[0,1,2,3,4,5])

        self.target_field = self.list_target_files[self.target_mode]


        self.best_loss = float('inf')
        inverse_fourier_field = self.ft_lens(self.target_field, pad=False, flag_ifft=True)





        # phase = torch.fmod(torch.angle(inverse_fourier_field),2*torch.pi)
        if target_mode_nb %2 == 0:
            phase = torch.angle(inverse_fourier_field)
            phase[torch.abs(phase) < torch.pi/2] = 0.000001
            phase[torch.abs(phase) > torch.pi/2] = torch.pi-0.000001
            phase_logit = -500*torch.log((torch.pi-phase)/phase)

        else:
            phase = torch.angle(inverse_fourier_field) + torch.pi/2
            # # if abs(phase) <0.01 replace with 0
            phase[torch.abs(phase) < torch.pi/2] = 0.000001
            phase[torch.abs(phase) > torch.pi/2] = torch.pi-0.000001
            #
            phase_logit = -500*torch.log((torch.pi-phase)/phase)




        self.slm = SLM(config_dict=self.config_slm, XY_grid=self.simulation.XY_grid)
        self.slm.amplitude.requires_grad = False

        self.out_field = None
        self.log_dir = log_dir
        self.writer = SummaryWriter(self.log_dir)

        if device == "gpu":
            device = "cuda"
        else:
            device = "cpu"
        self.weights_map = self.hypergaussian_weights(XY_grid=self.ft_lens.XY_output_grid, x_center=0, y_center=0, sigma_x=19*um, sigma_y=19*um, order=12).to(device)
        #
    def print_trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f'Trainable parameter: {name}')

    def forward(self, x):

        modulated_field = self.slm.apply_phase_modulation(x,beta=0.1,mapping=True,parity=self.target_mode % 2)

        # modulated_field = self.slm.apply_phase_modulation(x,beta=((self.current_epoch+1)/500+0),mapping=False,parity=self.target_mode % 2)


        # energy_input = torch.sum(torch.sum(torch.abs(x) ** 2))
        # modulated_field = self.slm.apply_amplitude_modulation(modulated_field)
        self.out_field = self.ft_lens(modulated_field, pad=False, flag_ifft=False)


        # energy_output = torch.sum(torch.sum(torch.abs(self.out_field) ** 2))


        return self.out_field

    def hypergaussian_weights(self, XY_grid, x_center, y_center, sigma_x, sigma_y, order):
        # Calculate distances from center
        x_grid = XY_grid[0]  # X coordinates of the grid
        y_grid = XY_grid[1]  # Y coordinates of the grid

        x_dist = (x_grid - x_center) ** 2 / (2 * sigma_x ** 2)
        y_dist = (y_grid - y_center) ** 2 / (2 * sigma_y ** 2)

        # Calculate hypergaussian weights
        weights_x = torch.exp(-x_dist ** order)
        weights_y = torch.exp(-y_dist ** order)

        # Produce a square-shaped weight map by multiplying independent x and y weights
        weight_map = weights_x * weights_y
        weight_map /= weight_map.sum()  # Normalize the weights to sum to 1
        return weight_map





    def compute_loss(self, out_field, list_target_field):

        energy_out = torch.sum(torch.sum(torch.abs(out_field) ))

        # Apply the weight map to the fields
        weighted_out_field = (out_field * self.weights_map)

        list_overlaps = []

        for idx,target_field in enumerate(list_target_field):

            energy_target = torch.sum(torch.sum(torch.abs(target_field)))
            target_field_norm = target_field  * energy_out/energy_target

            weighted_target_field = (target_field_norm * self.weights_map)
            overlap = calculate_normalized_overlap(weighted_out_field, weighted_target_field)
            list_overlaps.append(overlap)


        list_overlap_tmp = list_overlaps.copy()

        loss = -1 * torch.log(list_overlap_tmp[self.target_mode])


        list_overlap_tmp.pop(self.target_mode)


        loss += -10 * torch.log(1 - torch.max(torch.stack(list_overlap_tmp)))


        return loss, list_overlaps

    def training_step(self, batch, batch_idx):
        source_field, list_target_field = batch

        source_field = source_field.clone().detach().requires_grad_(True)  # Clone and detach
        out_field = self.forward(source_field)  # Directly use source field to ensure no input issues

        loss, list_overlap = self.compute_loss(out_field, list_target_field)

        self.current_loss = loss

        # Log gradients
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.writer.add_scalar(f'grad_norm/{name}', param.grad.norm(), self.global_step)

        for i, overlap in enumerate(list_overlap):
            self.log(f'list_overlap/mode_{i}', overlap, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log('source waist', self.source.sigma*2/mm, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def on_train_epoch_end(self, outputs=None) -> None:


        if self.current_epoch % 100 == 0:

            print("Saving the model")

            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                # Save the current best phase
                best_phase = self.slm.phase.cpu().detach().numpy()
                np.save(f'{self.log_dir}/best_slm_phase_{self.target_mode}.npy', best_phase)

            # Continue with your visualization and saving routines
            output_field = torch.abs(self.out_field.cpu().detach()).numpy()
            output_field_norm = (output_field - output_field.min()) / (output_field.max() - output_field.min())
            self.writer.add_image("output_field", output_field_norm, self.current_epoch)

            target_field = torch.abs(self.target_field.unsqueeze(0).cpu().detach()).numpy()
            target_field_norm = (target_field - target_field.min()) / (target_field.max() - target_field.min())
            self.writer.add_image("target_field", target_field_norm, self.current_epoch)

            # Current SLM phase
            slm_phase = self.slm.phase.cpu().detach().unsqueeze(0).numpy()
            slm_phase_norm = (slm_phase - slm_phase.min()) / (slm_phase.max() - slm_phase.min())
            self.writer.add_image("slm_phase", slm_phase_norm, self.current_epoch)

            # Current SLM phase
            quantified_phase = self.quantized_phase.cpu().detach().unsqueeze(0).numpy()
            quantized_phase_norm = (quantified_phase - quantified_phase.min()) / (quantified_phase.max() - quantified_phase.min())
            self.writer.add_image("quantized_phase_norm", quantized_phase_norm, self.current_epoch)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=0.5)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-1)
        return {"optimizer": optimizer}



    def train_dataloader(self):
        dataset = FixedDataset(self.source.field.field, self.list_target_files)
        return DataLoader(dataset, batch_size=1)




from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping



device = "gpu"


for i in range(8):

    if i%2==0:

        model = OpticalSystem(log_dir="./lightning_logs",device=device,target_mode_nb=i)
        model.print_trainable_parameters()

        tensorboard_logger = TensorBoardLogger("./lightning_logs", name="my_model")  # Specify the log directory

        early_stop_callback = EarlyStopping(
        monitor='train_loss',
        min_delta=0.00,
        patience=1000,  # Number of epochs with no improvement after which training will be stopped
        verbose=False,
        mode='min'
        )

        trainer = pl.Trainer(
        enable_progress_bar = True, # for turning off progress bar
        enable_model_summary= True, # for turning off weight summary.
        max_epochs          = 12000,
        enable_checkpointing= False,
        profiler            = None,
        logger              = None,
        accelerator=device,
        callbacks=[early_stop_callback],
        )
        trainer.fit(model)
    else:
        pass
