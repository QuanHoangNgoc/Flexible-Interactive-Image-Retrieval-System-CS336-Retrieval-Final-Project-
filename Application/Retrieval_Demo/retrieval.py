import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_ordered(query_enc, Mdb):
    """
    Computes cosine similarity between the encoded text and Mdb,
    and returns an ordered list of indices based on similarity.

    Args:
        text (str): The input text.
        encoder (SigLipEncoder): The encoder object.
        Mdb (np.ndarray): The database of encoded vectors.

    Returns:
        np.ndarray: An array of indices sorted by cosine similarity.
    """
    similarities = cosine_similarity(
        query_enc.reshape(1, -1), Mdb
    )  # Reshape enc_np for cosine_similarity
    ordered_indices = np.argsort(similarities[0])[::-1]  # Sort in descending order
    return ordered_indices


def representation(text, encoder):
    query_enc = encoder.get_np_text(text)[0]
    return query_enc


##########################################################################################
###          #############################################################################
##########################################################################################
def get_centroid(x):
    if x is None:
        return None
    if isinstance(x, list):
        return np.stack(x, dim=0).mean(0)
    else:
        # elif isinstance(x, np.) and x.ndim == 2:
        return np.mean(x, 0)


def rocchio_relevance_feedback(
    query=None, positive=None, negative=None, alpha=1, beta=0.8, gamma=0.1
):
    """
    Rocchio algorithm for relevance feedback as follows:
        newQuery = alpha * query + beta * centroid(positive) - gamma * centroid(negative)
    Args:
        query:
        positive:
        negative:
        alpha, beta, gamma:
    Returns:
        newQuery:
    """
    # print("Rocchio relevance feedback", end=" [>] ")

    newQuery = query * alpha
    if positive is not None:
        newQuery += beta * get_centroid(positive)
    if negative is not None:
        newQuery -= gamma * get_centroid(negative)
    return newQuery


def iterative_loop(text, encoder, Mdb, number=0):
    q = representation(text, encoder)
    order = get_ordered(q, Mdb)

    top_positive = 3
    for nth in range(number):
        q = rocchio_relevance_feedback(
            q, Mdb[order[:top_positive]], Mdb[order[-top_positive:]]
        )
        # q = rocchio_relevance_feedback(q, Mdb[order[:top_positive]]) # psuedo
        order = get_ordered(q, Mdb)
    return order


##########################################################################################
###          #############################################################################
##########################################################################################
def refine_loop(clicked, text, encoder, Mdb):
    q = representation(text, encoder)
    q = rocchio_relevance_feedback(q, Mdb[clicked], None)
    order = get_ordered(q, Mdb)
    return order


##########################################################################################
###          #############################################################################
##########################################################################################
def image_search_loop(pil_image, encoder, Mdb):
    q = encoder.get_np_image(pil_image)[0]
    order = get_ordered(q, Mdb)
    return order


def text_image_search_loop(text, pil_image, encoder, Mdb):
    full_order = iterative_loop(text=text, encoder=encoder, Mdb=Mdb, number=0)  # [0, N]
    sub_order = image_search_loop(
        pil_image=pil_image,
        encoder=encoder,
        Mdb=Mdb[full_order.tolist()[:1000]],  # [0, 1000]
    )
    full_order[:1000] = full_order[sub_order]
    return full_order
