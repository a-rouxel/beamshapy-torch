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
from ft_lens import ASMPropagation, create_gaussian_beam, create_binary_fresnel_lens, propagate_and_collect_side_view
from FieldGeneration import generate_target_profiles
from torch.utils.tensorboard import SummaryWriter
from cost_functions import *
import torch.nn.functional as F
from pytorch_lightning.profilers import AdvancedProfiler
import cProfile
import pstats
import io



class FixedDataset(Dataset):
    def __init__(self, source_field, list_target_field):
        self.source_field = source_field
        self.list_target_field = list_target_field

    def __len__(self):
        return 1  # Only one element in the dataset

    def __getitem__(self, idx):
        return self.source_field, self.list_target_field

class OpticalSystem(pl.LightningModule):

    def __init__(self,log_dir="./logs",device="cpu",target_mode_nb=2,try_nb=0,with_minimize_losses=None):

        super().__init__()
        self.config_source = load_yaml_config("./configs/source.yml")
        self.config_slm = load_yaml_config("./configs/SLM.yml")
        self.config_simulation = load_yaml_config("./configs/simulation_ASM.yml")

        self.simulation = Simulation(config_dict=self.config_simulation)
        self.source = Source(config_dict=self.config_source, XY_grid=self.simulation.XY_grid)
        # self.ft_lens = FT_Lens(self.simulation.delta_x_in, self.simulation.XY_grid, self.source.wavelength)
        self.asm = ASMPropagation(self.simulation.delta_x_in, self.source.wavelength, 20*mm, self.simulation.XY_grid[0].shape)

        self.target_mode = target_mode_nb
        self.with_minimize_losses = with_minimize_losses

        if self.target_mode %2 == 0:
            self.mode_parity = "even"
        else:
            self.mode_parity = "odd"


        self.list_target_files = generate_target_profiles(yaml_file="./configs/target_profile.yml",
                                                     XY_grid=self.simulation.XY_grid,
                                                     list_modes_nb=[0,1,2,3,4,5,6,7,8])

        self.target_field = self.list_target_files[self.target_mode]
        self.try_nb = try_nb

        self.best_loss = float('inf')


        self.slm = SLM(config_dict=self.config_slm, XY_grid=self.simulation.XY_grid)
        self.slm.amplitude.requires_grad = False

        self.out_field = None
        self.log_dir = log_dir
        self.writer = SummaryWriter(self.log_dir)

        if device == "gpu":
            device = "cuda"
        else:
            device = "cpu"
        self.weights_map = self.hypergaussian_weights(XY_grid=self.simulation.XY_grid, x_center=0, y_center=0, sigma_x=19*um, sigma_y=19*um, order=12).to(device)
        #
    def print_trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f'Trainable parameter: {name}')

    def forward(self, x):


        modulated_field = self.slm.apply_phase_modulation_sigmoid(x,steepness=2 + self.current_epoch/200,num_terms=1, spacing=1,mode_parity=self.mode_parity)

        self.out_field = self.asm(modulated_field)
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
        # weight_map /= weight_map.sum()  # Normalize the weights to sum to 1
        return weight_map



    def compute_loss(self, out_field, list_target_field):


        energy_out = torch.sum(torch.sum(torch.abs(out_field)**2 ))
        if energy_out == 0:
            energy_out += 1e-10  # Add a small number to avoid divide by zero

        # Apply the weight map to the fields
        weighted_out_field = (out_field * self.weights_map)

        self.inside_energy_percentage = torch.sum(torch.sum(torch.abs(weighted_out_field)**2))/energy_out

        list_overlaps = []

        for idx,target_field in enumerate(list_target_field):

            energy_target = torch.sum(torch.sum(torch.abs(target_field)))
            target_field_norm = target_field  * energy_out/energy_target

            weighted_target_field = (target_field_norm * self.weights_map)
            overlap = calculate_normalized_overlap(weighted_out_field, weighted_target_field)
            list_overlaps.append(overlap)


        list_overlap_tmp = list_overlaps.copy()

        loss1 = -1 * torch.log(list_overlap_tmp[self.target_mode] + 1e-10)  # Avoid log(0)

        list_overlap_tmp.pop(self.target_mode)


        loss2 = -1*(self.current_epoch) * torch.log(1 - torch.max(torch.stack(list_overlap_tmp)))


        # Calculate the differences along each dimension
        diff_dim0 = torch.diff(self.slm.phase, dim=0) ** 2
        diff_dim1 = torch.diff(self.slm.phase, dim=1) ** 2

        # Pad the last row and column of the difference tensors to match the original size
        diff_dim0_padded = F.pad(diff_dim0, (0, 0, 0, 1), 'constant', 0)
        diff_dim1_padded = F.pad(diff_dim1, (0, 1, 0, 0), 'constant', 0)


        # Now you can add the padded tensors together since they match in size
        TV_term = torch.sum(torch.sqrt(diff_dim0_padded + diff_dim1_padded + 1e-2))

        loss3 = TV_term*1e-5* (50/(self.current_epoch+1))

        
        # loss4 =  -1 * torch.log(self.inside_energy_percentage + 1e-10)
        loss4 = (1 - self.inside_energy_percentage) ** 2

        if self.with_minimize_losses:
            loss = loss1 + loss2 + loss3 + loss4
        else:
            loss = loss1 + loss2 + loss3


        return loss, loss1, loss2, loss3, loss4, list_overlaps
    
    def compute_loss_2(self, out_field, list_target_field):


        energy_out = torch.sum(torch.sum(torch.abs(out_field)**2 ))
        if energy_out == 0:
            energy_out += 1e-10  # Add a small number to avoid divide by zero

        # Apply the weight map to the fields
        weighted_out_field = (out_field * self.weights_map)

        self.inside_energy_percentage = torch.sum(torch.sum(torch.abs(weighted_out_field)**2))/energy_out

        list_overlaps = []

        for idx,target_field in enumerate(list_target_field):

            energy_target = torch.sum(torch.sum(torch.abs(target_field)))
            target_field_norm = target_field  * energy_out/energy_target

            weighted_target_field = (target_field_norm * self.weights_map)
            overlap = calculate_unnormalized_overlap(weighted_out_field, weighted_target_field)
            list_overlaps.append(overlap)


        list_overlap_tmp = list_overlaps.copy()

        loss1 = -1 * torch.log(list_overlap_tmp[self.target_mode] + 1e-10)*1e1  # Avoid log(0)


        loss2 = 0


        # Calculate the differences along each dimension
        diff_dim0 = torch.diff(self.slm.phase, dim=0) ** 2
        diff_dim1 = torch.diff(self.slm.phase, dim=1) ** 2

        # Pad the last row and column of the difference tensors to match the original size
        diff_dim0_padded = F.pad(diff_dim0, (0, 0, 0, 1), 'constant', 0)
        diff_dim1_padded = F.pad(diff_dim1, (0, 1, 0, 0), 'constant', 0)


        # Now you can add the padded tensors together since they match in size
        TV_term = torch.sum(torch.sqrt(diff_dim0_padded + diff_dim1_padded + 1e-2))

        loss3 = TV_term*1e-5* (50/(self.current_epoch+1))

        
        # loss4 =  -1 * torch.log(self.inside_energy_percentage + 1e-10)
        loss4 = 0

        if self.with_minimize_losses:
            loss = loss1 + loss2 + loss3 + loss4
        else:
            loss = loss1 + loss2 + loss


        return loss, loss1, loss2, loss3, loss4, list_overlaps

    def training_step(self, batch, batch_idx):
        source_field, list_target_field = batch

        source_field = source_field.clone().detach().requires_grad_(True)  # Clone and detach
        out_field = self.forward(source_field)  # Directly use source field to ensure no input issues

        loss, loss1, loss2, loss3, loss4, list_overlap = self.compute_loss(out_field, list_target_field)

        self.current_loss = loss

        # Log gradients
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         self.writer.add_scalar(f'grad_norm/{name}', param.grad.norm(), self.global_step)
        #
        for i, overlap in enumerate(list_overlap):
            self.log(f'list_overlap/mode_{i}', overlap, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log('inside_energy_percentage', self.inside_energy_percentage, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # self.log('source waist', self.source.sigma*2/mm, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss selectivity target ', loss1, on_step=True, on_epoch=True, logger=True)
        self.log('loss excitation others ', loss2, on_step=True, on_epoch=True, logger=True)
        self.log('loss TV', loss3, on_step=True, on_epoch=True, logger=True)
        self.log('loss energy inside', loss4, on_step=True, on_epoch=True, logger=True)
        return {'loss': loss}

    def on_after_backward(self):
        # Check for NaNs in all parameters
        for name, param in self.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN detected in gradients of {name} after backward")

    def on_train_epoch_end(self, outputs=None) -> None:


        if self.current_epoch >= 2000 and self.current_epoch % 250 == 0:

            print("Saving the model")

            # if self.current_loss < self.best_loss:
            #     self.best_loss = self.current_loss
            #     # Save the current best phase
            best_phase = self.slm.phase.cpu().detach().numpy()
            # np.save(f'{self.log_dir}/best_slm_phase_{self.target_mode}_{self.try_nb}_min_losses_{self.with_minimize_losses}.npy', best_phase)
            np.save(f'{self.log_dir}/slm_phase_{self.target_mode}_{self.try_nb}_min_losses_{self.with_minimize_losses}_epoch_{self.current_epoch}.npy', best_phase)

            # # Continue with your visualization and saving routines
            # output_field = torch.abs(self.out_field.cpu().detach()).numpy()
            # output_field_norm = (output_field - output_field.min()) / (output_field.max() - output_field.min())
            # self.writer.add_image("output_field", output_field_norm, self.current_epoch)

            # target_field = torch.abs(self.target_field.unsqueeze(0).cpu().detach()).numpy()
            # target_field_norm = (target_field - target_field.min()) / (target_field.max() - target_field.min())
            # self.writer.add_image("target_field", target_field_norm, self.current_epoch)

            # # Current SLM phase
            # slm_phase = self.slm.phase.cpu().detach().unsqueeze(0).numpy()
            # slm_phase_norm = (slm_phase - slm_phase.min()) / (slm_phase.max() - slm_phase.min())
            # self.writer.add_image("slm_phase", slm_phase_norm, self.current_epoch)

            # self.log_side_view_to_tensorboard()

    def log_side_view_to_tensorboard(self):


        source_field = self.source.field.field.to(self.device)  # Clone and detach
        best_phase = self.slm.phase


        u_in_masked = source_field * best_phase

        # Generate z values
        z_values = torch.linspace(0, 15*mm, 100)

        # Generate the side view intensity pattern
        side_view_intensity = propagate_and_collect_side_view(u_in_masked,self.simulation.delta_x_in, self.source.wavelength, z_values, self.simulation.XY_grid[0].shape)
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 6))
        extent = [z_values[0].item(), z_values[-1].item(), -self.simulation.XY_grid[0].shape[0]//2, self.simulation.XY_grid[0].shape[0]//2]
        im = ax.imshow(side_view_intensity.T, aspect='auto', extent=extent,origin='lower', cmap='hot')
        ax.set_title('Side View of Beam Propagation')
        ax.set_xlabel('z distance (m)')
        ax.set_ylabel('y position (pixels)')
        fig.colorbar(im, ax=ax, label='Normalized Intensity')

        # Log the figure to TensorBoard
        self.logger.experiment.add_figure('Side View Propagation', fig, self.current_epoch)
        plt.close(fig)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-2)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2, eta_min=1e-2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler1}
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

    #     # Define the first three cosine annealing schedulers with restarts
    #     scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-2)
    #     scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-2)
    #     scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-2)
        
    #     # Define a constant scheduler to maintain the learning rate after the restarts
    #     scheduler4 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)

    #     # Chain the schedulers together
    #     scheduler = torch.optim.lr_scheduler.SequentialLR(
    #         optimizer,
    #         schedulers=[scheduler1, scheduler2, scheduler3, scheduler4],
    #         milestones=[100, 300, 700]
    #     )

    #     return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def train_dataloader(self):
        dataset = FixedDataset(self.source.field.field, self.list_target_files)
        return DataLoader(dataset, batch_size=1)



from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping



if __name__ == "__main__":
    # Set up your model, trainer, etc. as before
    device = "gpu"


    # Your existing training loop
    for try_nb in range(3):
        list_minimize_losses = [True,False]
        for minimize_losses in list_minimize_losses:
            for i in range(9):

                model = OpticalSystem(log_dir="./lightning_logs", 
                                    device=device, 
                                    target_mode_nb=i, 
                                    try_nb=try_nb,
                                    with_minimize_losses=minimize_losses)

                tensorboard_logger = TensorBoardLogger("./lightning_logs", name="my_model")
                early_stop_callback = EarlyStopping(
                    monitor='train_loss',
                    min_delta=0.00,
                    patience=2000,
                    verbose=False,
                    mode='min'
                )

                trainer = pl.Trainer(
                    enable_progress_bar=True,
                    enable_model_summary=True,
                    max_epochs=8000,
                    enable_checkpointing=False,
                    profiler=None,
                    logger=None,
                    accelerator=device,
                    callbacks=[early_stop_callback],
                )
                trainer.fit(model)


