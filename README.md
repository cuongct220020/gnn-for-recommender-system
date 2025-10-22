# Applying GNNs to Steam Game Recommendation

> **IT3930 - Project II – Hanoi University of Science and Technology**


## 📌 Overview

This project explores the application of **Graph Neural Networks (GNNs)** for building a **collaborative filtering (CF)**-based recommendation system on the **Steam** platform. We compare four models: **Matrix Factorization (MF)**, **Neural Matrix Factorization (NeuMF)**, **Neural Graph Collaborative Filtering (NGCF)**, and **LightGCN**, using implicit user–game interaction data from [Kaggle](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam).

## 🧪 Environment & Dependencies

- **Language**: Python 3.13  
- **Key Libraries**:
  - **Data preprocessing & visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`
  - **Modeling & training**: `PyTorch`, `PyTorch Geometric`
- **Recommended runtime**: **Google Colab (with GPU enabled)**

> 💡 No local setup needed — just open the `.ipynb` files in Colab and run!

## 🚀 How to Run

1. Open any notebook in this repo (e.g., `LightGCN_Full_Corpus.ipynb`, `LightGCN_Leave_One_Last.ipynb`, etc.) in **Google Colab**.
2. Go to **Runtime → Change runtime type → Hardware accelerator: GPU**.
3. Run all cells sequentially. Dependencies will be installed automatically.
4. Processed data is saved in the `data/` folder and reused across notebooks.

## 🧹 Data Preprocessing (EDA Summary)

Starting from `recommendations.csv` (41M interactions), we applied:

1. **Missing data check**: No missing values found.
2. **Distribution analysis**: 
   - Most games have few reviews; a few are extremely popular (long-tail distribution).
   - Positive recommendations dominate (~85%).
3. **Outlier filtering**:
   - Keep only **games** with ≥ 1,000 reviews and released between **2009–2023**.
   - Keep only **users** with ≥ 30 reviews and at least one positive + one negative rating in the train set.
4. **Feature transformation**:
   - Convert `is_recommended` to binary (1/0).
   - Apply `log10(1 + hours)` to reduce skewness in playtime.
5. **Temporal split**: Sort by interaction date and split **80% train / 20% test** to prevent time leakage.

Final dataset: **2.53M interactions**, **99.1% sparsity**.

## 🤖 Models Implemented

All models use **BPR loss** and are optimized with **Adam**:

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Type</th>
      <th>Key Idea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>MF-BPR</strong></td>
      <td>Matrix Factorization</td>
      <td>Learns linear user/item embeddings via dot product.</td>
    </tr>
    <tr>
      <td><strong>NeuMF</strong></td>
      <td>Neural MF</td>
      <td>Combines GMF (linear) + MLP (non-linear) for richer interaction modeling.</td>
    </tr>
    <tr>
      <td><strong>NGCF</strong></td>
      <td>GNN-based CF</td>
      <td>Uses message passing with self-connections, linear transforms, and nonlinear activation.</td>
    </tr>
    <tr>
      <td><strong>LightGCN</strong></td>
      <td>Simplified NGCF</td>
      <td>Removes self-loops, feature transforms, and nonlinearities — only neighborhood aggregation remains.</td>
    </tr>
  </tbody>
</table>

**Shared hyperparameters**:
- `embedding_dim = 64`
- `batch_size = 8192`
- `epochs = 50`
- `learning_rate = 0.01` (with `CosineAnnealingLR`)
- `negative sampling ratio = 1:3`

## 📊 Evaluation Strategies

We evaluate using **top-10 metrics**: Precision@10, Recall@10, NDCG@10, HitRate@10.

### 1. **Full-corpus Evaluation**
- Split by **global timestamp**: all interactions before T → train, after T → test.
- Rank **all items** (except seen ones) for each user.
- ✅ Realistic, no time leakage  
- ❌ Computationally heavy; many cold-start cases

### 2. **Leave-one-last Evaluation**
- For each user: keep **last positive interaction** as test item.
- Rank it among **100 randomly sampled negatives**.
- ✅ Efficient; uses almost all data for training  
- ❌ May introduce minor time leakage; Precision/Recall less meaningful (only 1 relevant item)

## 📈 Results

Below are the evaluation results on both strategies. All scores are reported as mean values over multiple runs.
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th colspan="4" style="text-align: center;">Full-corpus Metrics</th>
      <th colspan="2" style="text-align: center;">Leave-one-last Metrics</th>
    </tr>
    <tr>
      <th></th>
      <th>Prec@10</th>
      <th>Rec@10</th>
      <th>NDCG@10</th>
      <th>Hit@10</th>
      <th>NDCG@10</th>
      <th>Hit@10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>MF-BPR</strong></td>
      <td>0.0111</td>
      <td>0.0123</td>
      <td>0.0143</td>
      <td>0.0927</td>
      <td>0.1614</td>
      <td>0.3151</td>
    </tr>
    <tr>
      <td><strong>NeuMF</strong></td>
      <td>0.0127</td>
      <td>0.0140</td>
      <td>0.0161</td>
      <td>0.1029</td>
      <td>0.1655</td>
      <td>0.3146</td>
    </tr>
    <tr>
      <td><strong>NGCF</strong></td>
      <td>0.0244</td>
      <td>0.0281</td>
      <td>0.0315</td>
      <td>0.1837</td>
      <td><strong>0.2685</strong></td>
      <td><strong>0.5165</strong></td>
    </tr>
    <tr>
      <td><strong>LightGCN</strong></td>
      <td><strong>0.0256</strong></td>
      <td><strong>0.0298</strong></td>
      <td><strong>0.0331</strong></td>
      <td><strong>0.1906</strong></td>
      <td>0.2681</td>
      <td>0.5069</td>
    </tr>
  </tbody>
</table>

> **Key Insight**:  
> GNN-based models (**NGCF**, **LightGCN**) significantly outperform traditional CF methods. **LightGCN** achieves the best performance on **Full-corpus**, indicating stronger generalization in real-world settings.

## 📚 Full Report

For detailed methodology, architecture diagrams, hyperparameter tuning, and analysis, please refer to the full report:  
📄 **[Report_Project2_Application_of_GNN_in_RecSys.pdf](Report_Project2_Application_of_GNN_in_RecSys.pdf)**

> 🔮 **Future Work**: Explore **UltraGCN**, which eliminates explicit message-passing layers and directly optimizes embeddings via ranking and constraint losses.

## 📝 License
Educational purpose

## 👨‍💻 Author
[@cuongct220020](https://github.com/cuongct220020)
