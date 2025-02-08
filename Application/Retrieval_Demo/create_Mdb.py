import numpy as np
from tqdm import tqdm
from PIL import Image 
import os 
import dataset, model 


def build_M_db(ds, encoder, batch_size=64):
    """
    Build the M database by encoding images in batches for faster processing.
    
    Args:
        ds: Dataset containing image data.
        encoder: Encoder object with a `get_np_image` method.
        batch_size: Number of images to process per batch.
    
    Returns:
        M: Numpy array of encoded image representations.
    """
    M = []

    # Preprocess data in batches
    for start_idx in tqdm(range(0, len(ds), batch_size), desc="Processing Batches"):
        # Get a batch of images
        batch = [ds[i] for i in range(start_idx, min(start_idx + batch_size, len(ds)))]

        # Encode the batch (assumes `get_np_image` can process a list of images)
        enc_np_batch = encoder.get_np_image(batch)  # Ensure encoder supports batch processing

        # Add encoded batch to M
        M.append(enc_np_batch)

    # Convert list of batches to a single NumPy array
    M = np.vstack(M)
    return M


source_data = [Image.open(path) for path in os.listdir(dataset.PATH)] 
Mdb = build_M_db(source_data, model.siglip) #[warn] long run
print(Mdb.shape)