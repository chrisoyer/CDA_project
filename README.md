# Team Members:
* Christopher Oyer

# Project Title: 
* Diverse Music recommendations from NonNegative Matrix Factorization 

# Problem Statement:
* I will create a recomendation engine for music based on on non-negative matrix factorization.
* I would like recommendations to be "novel", not "you'd like the Beatles, Adele, and Drake." 
* To make the engine adjustable: 
    * I will include an adjustable tuning parameter at prediction time for rare vs basic, so one doesn't just get stuff one has heard of and hasn't bothered to review.
    * use a mixture of tastes, so can pick from several taste clusters. E.g. one's favorite ambient music artist for studying and favorite hip-hop artist for workouts are not related, and it is useful to consider one user's taste to be the combination of several taste 'personas'. 
    * This intereacts with the data source I am using, which is made up of (user, song, play_count) tuples. The play_count will indicate how much the user likes the artists, so this will bias the system to recommend more-played artist, which will need counteracting.

# Data Source
I am using http://millionsongdataset.com/tasteprofile/, and aggregating from song counts up to artist counts. I think artist recommendations are more useful, and processing to artist counts will reduce the matrix sizes and be faster to train. 

# Methodology

### Factorization method: NNMF extention. 
I will be factoring a ratings matrix, $\textbf{X} \in \mathbb{R}^{m \times n} $  
as a feature matrix $\textbf{W} \in \mathbb{R}^{m \times k}$ and a user tensor $\textbf{U} \in \mathbb{R}^{m \times n \times p}  $ 
The P dimension is a set of several 'personas'. Each user is given several personas, which account for different types of taste they might have. 
with the factorization defined as

* $ \textbf{X}\equiv argmax _{p=1,2,...P}\textbf{U}_{iu}\textbf{V}_d^{\intercal} $  
i.e, for each user x latent feature combination, choose the persona that maximixes this dot product, then multiply the resulting matrix against the item matrix.




### Loss function:
* prediction accuracy: l2 of difference between predicted and observed (for those user/item pairs that actually existed)
* l2 regularization penalty for latent factors, to try to find denser representation space of latent factors. (may or may not include)
* l2 regularization for reccomendations: densify predictions, so heavily predicted items are penalized
* l2 regularization for number of latent factors in a user persona, to penalize 'single-persona' user state,
* penalty term on inverse of distance between each user's personas, to penalize representations that have similar personas within users: I would like to find diverse personas for each user. Penalty is sum over all distances between user's persona vectors. The penalty is 1/x^2. which is differentiable to -2/(x^3 * sum(norm derivatives)). I am concerned with two issues on this:
    * it doesn't go to zero so the analytical solution would be $\infty$. I am planning to subtract a small hyperparameter term to ensure this maxes out after reaching some size.
    * I am not sure I have the right derivative of an inverse of an l2 norm of a sum of l2 norms. If this is intractable I am hoping a good initialization is sufficient.


$loss \equiv \lambda_{predAcc}\sum_{i,j\in X} (\textbf{X}_{i,j} - \textbf{W}_{i,j}\textbf{F}_{i,j})^2 $  
        $+ \lambda_{predDiversity}\lVert\sum_u\textbf{W}_{i,j}\textbf{F}_{i,j} \rVert^2_2 $  
        $+\lambda_{personas}\sum_{p}\sum_{u}\lVert U_k \rVert_2^2 $  
        $- \lambda_{personaDistinction} \sum_m \dfrac{1}{\lVert \sum_{i\in p} \sum_{j\in p \;s.t.\; i \neq j}  (\lVert \textbf{U}_k \rVert_{2(i)}^2 - \lVert \textbf{U}_k \rVert_{2(j)}^2) \rVert_{2}^2}$  


### Preprocessing
could use counts, with 1 listen slightly negative to large counts = positive, like 
```python
    .assign(log_nos=lambda df: np.log2(np.where(df['numbers']==0, 3, df['numbers'])+.1)-2))
```
* Rational: many listens means you like something, but one listen means you tried and didn't like. 

### Initialization
* simple copy of user vector
* cluster items and assign user vectors to top x clusters they are closest to.

### prediction
* for prediction (not during training), add a multiplication by a scalar & matrix for 'rarety' of item. 
    * Rarety would be the overall frequency of an item.  
    * The additional term would be, for a prediction vector for user 'u' $\textbf{p} = \hat{\textbf{X}}_u$  and item frequency vector  $\textbf{freq}$ and item inverse frequency vector $\textbf{invfreq}_n \equiv  1/\textbf{freq}_n$   
the prediction would be $\textbf{p} \odot (tuningFactor* \textbf{freq}) \odot \textbf{invfreq}$  
    * So using the tuning scalar with tuningFactor=1 would be the same; tuningFactor>1 would favor common items, and tuningFactor < 1 would favor rare items.

### Training algorithm
* Assuming I did the math correctly, I will be using Alternating Least Squares with numpy, which seems to have fast convergence. If there is a problem, I will use SGD, which would probably be best implemented in a neural network.
* ALS is defined as iteratively setting $\textbf{W}$ or $\textbf{F}$ as a constant, and solving the $\textbf{X} = \textbf{W}\textbf{F}$ equation with an analytical solution, then switching the free and fixed variable, solving again, until convergence.

## Evaluation and Final Results

* I will look at the loss as defined above; the reconstruction loss will be the most important; the other regularization terms will hopefully allows qualitatively good predictions while being at worst neutral in their effect on reconstruction loss. 
* I will look at the prediction results across as holdout set, both for accuracy, and separately for diversity: as an important goal of this reccommender is to give recommendations that are new, so statistical measure like 'fit exponential distribution to prediction frequencies and favor results with low $\lambda$ vaule. 
* Hyperparameter tuning as well as testing actual recommendations will be used.

### Prior work/ references:
* https://dl.acm.org/doi/pdf/10.1145/2507157.2507209
* https://cseweb.ucsd.edu/~jmcauley/pdfs/cikm15.pdf
