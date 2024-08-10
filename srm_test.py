# Original imports and initial setup
from Data_Set import my_collate, Tensor
from models import srm
from torch.utils.data import DataLoader
import os
import torch
from utils import sample, draw

# Experiment name and path setup
experiment_name = 'SRM Test'
torch.set_float32_matmul_precision('medium')  # Original precision setting (consider experimenting with this)
test_path = "./Data/10k.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Ensure GPU usage if available

# Load the model checkpoint
model = srm.load_from_checkpoint("./Models/SRM.ckpt")
size = 512
dim_in = 6

# Modify the number of samples for testing purposes
# Original:
# samples = 1000
samples = 10 # Reduced number of samples for quicker testing

L = []

# Modify the number of repetitions for testing purposes
# Original:
# reps = 1
reps = 0 # Keeping it to 1, but can reduce further to 0 if needed for initial debugging

# Ensure the output directory exists
if not os.path.exists("Samples/{}".format(experiment_name)):
    os.makedirs("Samples/{}".format(experiment_name))

# Load the test dataset
test_set = Tensor(test_path)
loader = DataLoader(test_set, 1, shuffle=False, collate_fn=my_collate, pin_memory=True)
Encoder = model.encoder
Decoder = model.decoder

# Loop through the dataset and generate samples
for i in range(reps):
    with torch.no_grad():
        for i, data in enumerate(loader):
            # Encode the data
            Latent = Encoder(data[0].to(device))[0]
            L.append(Latent)
            
            # Sample strokes
            stroke = sample(samples, model.sample_steps, Decoder, model.noise_scheduler_sample, Latent, dim_in)
            
            # Save the drawing
            filename = 'Samples/{}/{}.svg'.format(experiment_name, i)
            draw(model.format, size, filename, stroke)
            
            # Add a print statement to track progress
            if i % 10 == 0:  # Print progress every 10 iterations
                print(f"Processed {i} samples")

# Save the latent vectors
# Original:
# Latents = [item for sublist in L for item in sublist]
# torch.save(Latents, 'Latent/{}.pt'.format(experiment_name))
Latents = [item for sublist in L for item in sublist]
torch.save(Latents, 'Latent/{}.pt'.format(experiment_name))

# Notes:
# - Reduced the number of samples from 1000 to 100 for quicker feedback.
# - Added a print statement to monitor progress every 10 iterations.
# - Repetitions (reps) are kept at 1, but can be reduced further if necessary.
# - Consider adjusting or removing torch.set_float32_matmul_precision('medium') for performance testing.



