import streamlit as st
from PIL import Image
import controler

# Dummy image database
image_database = controler.dataset.image_database


# Function to retrieve and resize images to a uniform size
def retrieve_images(caption=None, uploaded_image=None, num_results=10, order=None):
    """
    Retrieves images relevant to a caption or uploaded image and resizes them to a uniform size.
    Args:
        caption (str): Input caption to search for.
        uploaded_image (Image): Uploaded image for retrieval.
        num_results (int): Number of results to retrieve.
        order (list): Predefined order of image indices.
    Returns:
        list of tuples: (Index, Path, Resized Image).
    """
    if order is None:
        if caption and uploaded_image:
            order = controler.text_image_search_loop(
                text=caption,
                pil_image=uploaded_image,
                encoder=controler.model.siglip,
                Mdb=controler.dataset.Mdb,
            )
        elif caption:
            order = controler.iterative_loop(
                caption, controler.model.siglip, controler.dataset.Mdb, 0
            )
        elif uploaded_image:
            order = controler.image_search_loop(
                uploaded_image, controler.model.siglip, controler.dataset.Mdb
            )
        else:
            st.error("Please provide a caption or upload an image for retrieval.")
            return []

    order = order[:num_results].tolist()

    results = []
    isfirst = True and controler.SEGMENT 
    for idx in order:
        path = image_database[idx]
        try:
            image = resize_image_from_path(path, caption or "", isfirst=isfirst)
            results.append((idx, path, image))
        except Exception as e:
            st.error(f"Error processing path {path}: {e}")
        isfirst = False

    return results


# Helper function to fetch and resize an image
def resize_image_from_path(full_path, text, isfirst, size=(224, 224)):
    image = Image.open(full_path)
    if isfirst and text:
        return controler.segment.segmenter.get_segment_image(image, text)
    return image.resize(size, Image.Resampling.LANCZOS)


# Display images in a grid with checkboxes
def display_images_with_checkboxes(results, columns=5):
    """
    Displays images in a grid with checkboxes.
    Args:
        results (list): List of (Index, Path, Resized Image) tuples.
        columns (int): Number of images per row.
    Returns:
        list of selected indices based on checkboxes.
    """
    selected_indices = []
    selected_pils = []
    cols = st.columns(columns)

    for i, (idx, path, image) in enumerate(results):
        with cols[i % columns]:
            st.image(image, caption=f"Image {idx}", use_container_width=True)
            if st.checkbox(f"Select {idx}", key=f"select_{idx}"):
                selected_indices.append(idx)
                selected_pils.append(image)

    return selected_indices, selected_pils


# Main application logic
refine_cnt = 0


def main():
    st.set_page_config(page_title="Image Retrieval", layout="wide")
    st.title("🌟 Content Based Image Retrieval 🌟")

    st.markdown("---")
    st.markdown(
        """
        ## Instructions
        1. Enter a caption or upload an image to retrieve relevant images.
        2. Use checkboxes to select images that related with your needs.
        3. Click "Iterative" to refine results based on selections.
        4. Repeat the process as needed to narrow down results.
        """
    )

    # Initialize session state for storing results and selected images
    if "current_results" not in st.session_state:
        st.session_state["current_results"] = []
    if "selected_indices" not in st.session_state:
        st.session_state["selected_indices"] = []

    # Input for the caption
    caption_query = st.text_input("Enter your caption:")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
    uploaded_image = Image.open(uploaded_file) if uploaded_file else None

    # Search Button: Trigger a new search
    if st.button("Search"):
        if caption_query or uploaded_image:
            # Fetch results for the given caption or uploaded image
            st.session_state["current_results"] = retrieve_images(
                caption=caption_query, uploaded_image=uploaded_image
            )
            st.session_state["selected_indices"] = []  # Reset selections
        else:
            st.error("Please provide a caption or upload an image to search.")

    # Display current results
    if st.session_state["current_results"]:
        st.subheader("Results")
        # Display images and capture user selections
        selected_indices, selected_pils = display_images_with_checkboxes(
            st.session_state["current_results"]
        )

        # Update session state with the selected indices
        if selected_indices:
            st.session_state["selected_indices"] = selected_indices

        # Iterative Button: Refine the results
        if st.button("Iterative"):
            # Filter the results based on the selected images
            order = controler.refine_loop(
                selected_indices,
                caption_query,
                controler.model.siglip,
                controler.dataset.Mdb,
            )
            st.session_state["current_results"] = retrieve_images(
                caption=caption_query, order=order
            )

            # Preserve previous selections across iterations
            st.session_state["selected_indices"] = selected_indices

            global refine_cnt
            refine_cnt += 1
            st.markdown(
                f"""
                Success Refine {refine_cnt}! 
                """
            )

            st.markdown(
                f"""
                Query: '{caption_query}' logging! 
                """
            )

            #!!![warning] LLM feature
            if controler.LLM_FLAG:
                st.markdown(
                    controler.llms.edit_query(
                        clicked_pils=selected_pils, text=caption_query
                    )
                )


if __name__ == "__main__":
    main()
