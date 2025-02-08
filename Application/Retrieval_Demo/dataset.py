import numpy as np


# Load the saved Mdb file
Mdb = np.load("Mdb.npy")
print("Mdb has been loaded:", Mdb.shape)


PATH = "D:\cd_data_C\Downloads\images"


image_database = []
for i in range(0, len(Mdb)):
    full_path = PATH + "/" + f"image_{i}.jpg"
    full_path = full_path.replace("\\", "/")
    image_database.append(full_path)
