import torch

def calculate_normalized_overlap(field1, field2):
    """
    Calculate the normalized overlap integral of two complex fields using PyTorch.

    Parameters:
    - field1 (torch.Tensor): First complex field.
    - field2 (torch.Tensor): Second complex field.

    Returns:
    - float: The normalized overlap integral.
    """
    # Ensure inputs are complex tensors
    if not (torch.is_complex(field1) and torch.is_complex(field2)):
        raise ValueError("Input fields must be complex tensors.")

    # Compute the conjugates
    field1_conj = torch.conj(field1)
    field2_conj = torch.conj(field2)

    # Compute the overlap integral
    # overlap_integral = 0.5 * (torch.sum(field1 * field2_conj) + torch.sum(field1_conj * field2))
    overlap_integral = (torch.sum(field1 * field2_conj) * torch.sum(field1_conj * field2))
    # Compute the individual integrals of the absolute squares of both fields
    integral_field1 = torch.sum(torch.abs(field1) ** 2)
    integral_field2 = torch.sum(torch.abs(field2) ** 2)

    # Normalize the overlap integral
    # normalized_overlap = overlap_integral / torch.sqrt(integral_field1 * integral_field2)
    normalized_overlap = overlap_integral / (integral_field1 * integral_field2)

    return torch.abs(normalized_overlap)

def calculate_unnormalized_overlap(field1, field2):
    """
    Calculate the normalized overlap integral of two complex fields using PyTorch.

    Parameters:
    - field1 (torch.Tensor): First complex field.
    - field2 (torch.Tensor): Second complex field.

    Returns:
    - float: The normalized overlap integral.
    """
    # Ensure inputs are complex tensors
    if not (torch.is_complex(field1) and torch.is_complex(field2)):
        raise ValueError("Input fields must be complex tensors.")

    # Compute the conjugates
    field1_conj = torch.conj(field1)
    field2_conj = torch.conj(field2)

    # Compute the overlap integral
    # overlap_integral = 0.5 * (torch.sum(field1 * field2_conj) + torch.sum(field1_conj * field2))
    overlap_integral = (torch.sum(field1 * field2_conj) * torch.sum(field1_conj * field2))*1e-10
    # Compute the individual integrals of the absolute squares of both fields
    integral_field1 = torch.sum(torch.abs(field1) ** 2)
    integral_field2 = torch.sum(torch.abs(field2) ** 2)

    # Normalize the overlap integral
    # normalized_overlap = overlap_integral / torch.sqrt(integral_field1 * integral_field2)
    normalized_overlap = overlap_integral / (integral_field1 * integral_field2)

    return torch.abs(overlap_integral)

import matplotlib.pyplot as plt
def quantize_phase(phase_tensor, n_levels,mode_parity="even"):



    if mode_parity == "even":
        phase_tensor = phase_tensor
    elif mode_parity == "odd":
        phase_tensor = phase_tensor - torch.pi/2

    pi_tensor = torch.tensor(2 * torch.pi, dtype=phase_tensor.dtype, device=phase_tensor.device)
    # Calculate the step size for each quantization level
    step_size = pi_tensor / n_levels

    # Use torch.round to find the nearest level
    nearest_level = torch.round(phase_tensor / step_size)

    # Quantize the phase by computing the nearest level for each element
    quantized_phase = nearest_level * step_size

    if mode_parity == "even":
        quantized_phase = quantized_phase
    elif mode_parity == "odd":
        quantized_phase = quantized_phase + torch.pi/2


    return quantized_phase


def compute_loss_1(out_field, list_target_field,weights_map,target_mode):

        energy_out = torch.sum(torch.sum(torch.abs(out_field) ))

        # Apply the weight map to the fields
        weighted_out_field = (out_field * weights_map)

        list_overlaps = []

        for idx,target_field in enumerate(list_target_field):

            energy_target = torch.sum(torch.sum(torch.abs(target_field)))
            target_field_norm = target_field  * energy_out/energy_target

            weighted_target_field = (target_field_norm * weights_map)

            overlap = calculate_normalized_overlap(weighted_out_field, weighted_target_field)
            list_overlaps.append(overlap)


        list_overlap_tmp = list_overlaps.copy()

        loss = -1 * torch.log(list_overlap_tmp[target_mode])

        list_overlap_tmp.pop(target_mode)

        loss += -10 * torch.log(1 - torch.max(torch.stack(list_overlap_tmp)))

        return loss, list_overlaps

def compute_loss_discretization(self, out_field, list_target_field,n_levels=4):

    energy_out = torch.sum(torch.sum(torch.abs(out_field) ))

    # Apply the weight map to the fields
    weighted_out_field = (out_field * self.weights_map)

    list_overlaps = []
    list_overlap_tmp = []
    loss = 0




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


    self.quantized_phase = quantize_phase(self.slm.phase, n_levels)
    print(torch.unique(self.quantized_phase))

    rmse = torch.sqrt(torch.mean((self.slm.phase - self.quantized_phase) ** 2))


    binarization_term =  100*rmse

    print("current loss: ", loss)
    print("binarization term: ", binarization_term)

    loss += binarization_term

    return loss, list_overlaps


def generalized_sigmoid_function(x, steepness=20, num_terms=3, spacing=0.5):

    # Calculate shifts
    if num_terms % 2 == 0:
        shifts = torch.linspace(-spacing * (num_terms // 2 - 0.5), spacing * (num_terms // 2 - 0.5), num_terms)
    else:
        shifts = torch.linspace(-spacing * (num_terms // 2), spacing * (num_terms // 2), num_terms)

    # Initialize the result tensor to zero
    result = torch.zeros_like(x)

    # Calculate the sigmoid for each shift
    for shift in shifts:
        result += stable_sigmoid(x, shift,steepness=steepness, clip_range=10)

    # Normalize to expected maximum
    expected_max_result = 2 * torch.pi * (1 - 1 / (num_terms+1))
    max_result = torch.max(result)
    result = result * expected_max_result / max_result

    return result


def stable_sigmoid(x, shift,steepness=20, clip_range=10):
    # Clip the scaled input to prevent extreme values in the exponential function
    scaled_x = steepness * (x - shift)
    clipped_x = torch.clamp(scaled_x, -clip_range, clip_range)

    # Compute the sigmoid using the stable, clipped input
    sigmoid = 1 / (1 + torch.exp(-clipped_x))

    return sigmoid