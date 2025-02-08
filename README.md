# 🔍 **Flexible Interactive Retrieval System (QT-FIRS) for Content-Based Image Retrieval**

## 📌 What is it?
QT-FIRS (Flexible Interactive Retrieval System) is an advanced Content-Based Image Retrieval (CBIR) system designed to retrieve images based on textual queries. The project integrates state-of-the-art AI models (CLIP & SigLIP). It introduces an interactive, scalable, and user-friendly framework for improved retrieval accuracy.

## ❓ Why this project?
Traditional CBIR systems face challenges such as **scalability, interpretability, and interactivity**. QT-FIRS enhances the retrieval process by:
- 🌍 **Bridging the gap** between text and image representations using multimodal models.
- 🔄 **Improving interactivity** via feedback loops and iterative query refinement.
- ⚡ **Scaling efficiently** using **semantic hashing and collection expansion**.
- 🧠 **Leveraging Large Language Models (LLMs)** for better query understanding.

## 👥 Who can benefit?
- 🏢 **Multimedia & E-commerce platforms** – Enhance search experiences for visual content.
- 🎓 **Researchers & AI Enthusiasts** – Experiment with multimodal retrieval techniques.
- 📊 **Data Scientists & Developers** – Build scalable image retrieval applications.
- 🖼️ **Artists & Content Creators** – Discover similar images more effectively.

## 📊 Demo and Results
### **🔬 Evaluated on:**
- **MSCOCO & Flickr30k datasets** (36,014 images)
- **CLIP & SigLIP Baselines** (Comparing retrieval effective)
- **Enhanced techniques:**
  - 🔄 Iterative retrieval (Rocchio Algorithm)
  - 📖 LLM-based query refinement
  - ⚡ Semantic hashing for large-scale datasets
  - 📈 Collection expansion for dynamic datasets

### **🏆 Key Performance Metrics:**
| Method | R@1 | R@5 | R@10 | Mean Rank |
|--------|------|------|-------|------------|
| **CLIP (No Iteration)** | 23.10% | 43.66% | 53.17% | 196.76 |
| **CLIP (Iteration 1)** | 26.16% | 40.16% | 45.80% | 431.92 |
| **SigLIP (No Iteration)** | 40.36% | 62.77% | 70.98% | 107.10 |
| **SigLIP (Iteration 1)** | 43.98% | 60.23% | 66.48% | 133.51 |

### **📌 Key Findings:**
✅ **SigLIP outperforms CLIP** in retrieval accuracy.  
✅ **Iterative refinement improves retrieval precision**.  
✅ **Semantic hashing reduces storage by up to 64x and maintaining efficiency**.  

## 🛠️ How did we do it?
### **1️⃣ Contrastive Learning for Image Retrieval**
- 📖 **Text Encoder**: Transforms queries into vector embeddings.
- 🖼️ **Image Encoder**: Converts images into a shared embedding space.
- 📏 **Cosine Similarity Matching**: Finds the closest images based on the query.

### **2️⃣ Interactive Query Refinement**
- 🔄 **Iterative Retrieval with Rocchio Algorithm**: Adjusts query embeddings based on positive & negative feedback.
- 📖 **LLMs for Query Refinement**: Uses **Gemini 1.5 Flash** to improve textual queries.
- 📊 **Two-Phase Multimedia Queries**: Combines images + text for enhanced search results.

### **3️⃣ Scaling with Semantic Hashing & Collection Expansion**
- ⚡ **Semantic Hashing**: Compresses embeddings into binary vectors for fast retrieval.
- 📈 **Collection Expansion**: Dynamically adds textual descriptions to the image dataset.

## 📚 Key Learnings
- **Contrastive learning** effectively bridges text-image relationships.
- **User feedback loops** refine search accuracy over iterations.
- **Semantic hashing** is crucial for handling large-scale datasets efficiently.
- **LLMs improve query understanding**, but computational cost is a challenge.

## 🏅 Achievements
🎖️ **Developed a modular, scalable CBIR system** for real-world applications.  
📊 **Compared CLIP & SigLIP on a large dataset** with insightful results.  
🚀 **Integrated LLMs for query refinement**, making retrieval more user-friendly.  

## ✨ Author - Support & Contributions
👤 **Authors:** Quan Hoang Ngoc, Dai-Truong Le-Trong  
🏫 **Professor:** Thanh Duc Ngo (UIT - CS336)  
📩 **Contributions Welcome!** Fork the repo and experiment with retrieval enhancements!  
⭐ **Support:** If you find this project useful, give it a **star** ⭐ on GitHub!  

🚀 **Let's build smarter and more flexible image retrieval systems!** 🖼️
