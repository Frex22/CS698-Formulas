Here are the solutions to the exercises based on the provided graph and the concepts from Chapter 15.<img width="499" height="1384" alt="image" src="https://github.com/user-attachments/assets/9f0a1c78-1eec-4d59-93e7-338d452600ff" />



---

### 1. Graph Representation and BFS Trace

#### a. Adjacency Matrix
A $8 \times 8$ matrix where `1` indicates a directed edge from Row to Column.

$$
\begin{bmatrix}
0 & 0 & 1 & 0 & 0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 & 1 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 1 & 0 & 1 & 0 
\end{bmatrix}
$$

#### b. CSR Format
*   **`numVertices`**: 8
*   **`numEdges`**: 15
*   **`dst`**: `[2, 5, 0, 4, 7, 3, 0, 6, 3, 1, 7, 4, 2, 4, 6]`
*   **`srcPtrs`**: `[0, 2, 5, 6, 8, 9, 11, 12, 15]`

---

#### c. Parallel BFS Execution (Start Vertex: 0)

**The BFS Levels:**
*   **Level 0:** {0}
*   **Level 1:** Neighbors of 0 $\to$ **{2, 5}**
*   **Level 2:** Neighbors of 2 (3) and 5 (1, 7) $\to$ **{1, 3, 7}**
*   **Level 3:** Unvisited neighbors of 1 (4), 3 (6), 7 (4, 6) $\to$ **{4, 6}**
*   **Level 4:** Neighbors of 4 (3-visited), 6 (4-visited) $\to$ **{} (Done)**

**i. Vertex-Centric PUSH (Naive)**
*Assumes grid size = number of vertices ($N=8$).*

| Iteration | 1. Threads Launched | 2. Threads Iterating (Active) |
| :--- | :--- | :--- |
| **Iter 1 (L0 $\to$ L1)** | 8 | 1 (Thread 0) |
| **Iter 2 (L1 $\to$ L2)** | 8 | 2 (Threads 2, 5) |
| **Iter 3 (L2 $\to$ L3)** | 8 | 3 (Threads 1, 3, 7) |
| **Iter 4 (L3 $\to$ L4)** | 8 | 2 (Threads 4, 6) |

**ii. Vertex-Centric PULL (Naive)**
*Assumes grid size = number of vertices ($N=8$). Only unvisited threads iterate.*

| Iteration | 1. Threads Launched | 2. Threads Iterating (Unvisited) | 3. Threads Labeling (Success) |
| :--- | :--- | :--- | :--- |
| **Iter 1** | 8 | 7 (Nodes 1-7) | 2 (Nodes 2, 5 find parent 0) |
| **Iter 2** | 8 | 5 (Nodes 1,3,4,6,7) | 3 (Nodes 1, 3, 7 find parents) |
| **Iter 3** | 8 | 2 (Nodes 4,6) | 2 (Nodes 4, 6 find parents) |
| **Iter 4** | 8 | 0 (All visited) | 0 |

**iii. Edge-Centric**
*Assumes grid size = number of edges ($E=15$). A thread labels if Src is Active AND Dst is Unvisited.*

| Iteration | 1. Threads Launched | 2. Threads that *attempt* to Label |
| :--- | :--- | :--- |
| **Iter 1** | 15 | 2 (Edges $0 \to 2, 0 \to 5$) |
| **Iter 2** | 15 | 3 (Edges $2 \to 3, 5 \to 1, 5 \to 7$) |
| **Iter 3** | 15 | 4 (Edges $1 \to 4, 3 \to 6, 7 \to 4, 7 \to 6$) |
| **Iter 4** | 15 | 0 |

**iv. Vertex-Centric Push (Frontier-Based)**
*Grid size = Size of Frontier Queue.*

| Iteration | 1. Threads Launched | 2. Threads Iterating |
| :--- | :--- | :--- |
| **Iter 1** | 1 | 1 (Queue: {0}) |
| **Iter 2** | 2 | 2 (Queue: {2, 5}) |
| **Iter 3** | 3 | 3 (Queue: {1, 3, 7}) |
| **Iter 4** | 2 | 2 (Queue: {4, 6}) |

---

### 2. Direction-Optimized BFS Host Code

This implementation switches between Push and Pull based on the size of the frontier.

```cpp
void bfs_direction_optimized(CSRGraph csr, CSCGraph csc, int numVertices, 
                             int startVertex, int* level) {
    
    // 1. Initialization
    int* d_level;
    int* d_finished; // Flag to check if work was done
    cudaMalloc(&d_level, numVertices * sizeof(int));
    // Initialize levels to UINT_MAX, startVertex to 0
    // ... (cudaMemset/Memcpy logic omitted for brevity)

    int currLevel = 1;
    int frontierSize = 1; // Start with root
    int edgesToCheck = csr.srcPtrs[startVertex + 1] - csr.srcPtrs[startVertex];
    
    // Heuristic Thresholds (e.g., if > 10% of edges are active)
    int alpha = 10; 
    
    // 2. The Main Loop
    while (frontierSize > 0) {
        
        // --- DECISION LOGIC ---
        // If the active frontier connects to many edges, 
        // switching to PULL (bottom-up) is likely faster.
        if (edgesToCheck > csr.numEdges / alpha) {
            
            // LAUNCH PULL (Bottom-Up)
            // Uses CSC Graph. Launches N threads.
            int numBlocks = (numVertices + 255) / 256;
            bfs_kernel_pull<<<numBlocks, 256>>>(csc, d_level, d_finished, currLevel);
            
        } else {
            
            // LAUNCH PUSH (Top-Down)
            // Uses CSR Graph. Can use Frontier Queue or Naive N threads.
            // (Assuming Frontier Queue version for efficiency)
            int numBlocks = (frontierSize + 255) / 256;
            bfs_kernel_push_frontier<<<numBlocks, 256>>>(csr, d_level, 
                                                         d_currentFrontier, ...);
        }
        
        // 3. Maintenance
        // Update currLevel
        // Calculate new frontierSize and edgesToCheck for the next iteration
        // (This usually requires a reduction or atomic counter on GPU)
        currLevel++;
        // ... update logic
    }
}
```

---

### 3. Single-Block BFS Kernel Implementation

This kernel (Section 15.7) keeps the BFS inside one thread block (using Shared Memory) to avoid the overhead of launching new grids for small frontiers.

```cpp
__global__ void bfs_single_block(CSRGraph graph, int* level, 
                                 int* globalFrontier, int* globalFrontierCount,
                                 int currentLevel) {
    
    // 1. Shared Memory Queues
    __shared__ int sharedQ[1024]; 
    __shared__ int qCount;
    __shared__ int nextQCount;
    
    // 2. Load Global Frontier into Shared Queue (Initial Step)
    if (threadIdx.x == 0) {
        qCount = *globalFrontierCount;
        nextQCount = 0;
    }
    __syncthreads();
    
    // Only proceed if the frontier fits in shared memory
    if (qCount > 1024) return; // Fallback to global kernel
    
    // Parallel Load
    for (int i = threadIdx.x; i < qCount; i += blockDim.x) {
        sharedQ[i] = globalFrontier[i];
    }
    __syncthreads();

    // 3. The Persistent Loop
    // Keep iterating levels as long as queue is not empty and fits in SM
    while (qCount > 0 && qCount < 512) { // 512 leaves room for next level expansion
        
        int vertex = -1;
        if (threadIdx.x < qCount) {
            vertex = sharedQ[threadIdx.x];
        }
        
        __syncthreads(); // Wait for everyone to read their vertex
        
        // Reset count for the *next* level
        if (threadIdx.x == 0) qCount = 0; 
        __syncthreads();
        
        // Process neighbors
        if (vertex != -1) {
            for (int edge = graph.srcPtrs[vertex]; 
                 edge < graph.srcPtrs[vertex+1]; ++edge) {
                
                int neighbor = graph.dst[edge];
                
                // Atomic Check & Set
                if (atomicCAS(&level[neighbor], UINT_MAX, currentLevel) == UINT_MAX) {
                    // Add to Shared Queue
                    int idx = atomicAdd(&qCount, 1);
                    if (idx < 1024) {
                        sharedQ[idx] = neighbor;
                    } else {
                        // Overflow handling (set flag to switch to global kernel)
                    }
                }
            }
        }
        currentLevel++;
        __syncthreads(); // Wait for new queue to be built
    }
    
    // 4. Write back to Global Memory (if we exited loop)
    // ... Copy sharedQ to globalFrontier ...
}
```

---

### BFS Implementation Cheat Sheet

| Feature | **Push (Top-Down)** | **Pull (Bottom-Up)** | **Edge-Centric** |
| :--- | :--- | :--- | :--- |
| **Perspective** | "I tell my neighbors." | "I ask my parents." | "I check this connection." |
| **Active Thread** | Thread assigned to **Previous Level** node. | Thread assigned to **Unvisited** node. | Thread assigned to **Edge** $(u, v)$. |
| **Data Format** | **CSR** (`srcPtrs`, `dst`) | **CSC** (`dstPtrs`, `src`) | **COO** (`src`, `dst` pairs) |
| **Memory Write** | Writes to neighbor (`level[neighbor]`). | Writes to self (`level[me]`). | Writes to destination (`level[v]`). |
| **Contention?** | **Yes** (Multiple parents push to one child). | **No** (Only I update myself). | **Yes**. |
| **Optimization** | **Frontiers:** Only launch active threads. | **Early Exit:** Stop checking parents after finding one active parent. | Good load balance for "Celebrity" graphs. |
| **Best For...** | Sparse frontiers (Beginning/End of BFS). | Dense frontiers (Middle of BFS). | Extremely irregular graphs. |
