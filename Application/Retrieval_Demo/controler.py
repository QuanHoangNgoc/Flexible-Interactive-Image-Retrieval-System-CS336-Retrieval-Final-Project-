import dataset
import model
from retrieval import (
    iterative_loop,
    refine_loop,
    image_search_loop,
    text_image_search_loop,
)
SEGMENT = False 
if SEGMENT: 
    import segment


LLM_FLAG = False
if LLM_FLAG:
    import llms


if __name__ == "__main__":
    print(dataset.Mdb.shape)
    print(model.siglip)
