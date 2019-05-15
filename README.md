# feature-selection-search
Traverses a search tree to find the optimal feature set using K-nearest-neighbors with leave-one-out cross-validation.

# Datasets
small_data.txt: 200 instances with 10 features each
large_data.txt: 200 instances with 100 features each

# Algorithms
Greedy forward selection: Starts with an empty set. Evaluates all features at every level and adds the feature with 
the highest accuracy to the set.

Greedy backward selection: Same as above, except starts with a set of all the features and gradually removes them.

Optimized forward selection: Calculates the maximum amount of correct classifications over all the previous levels.
Using this number, it calculates the maximum amount of incorrect classifications it can make at the current level 
without doing worse than the current best. If it surpasses this limit, it simply ends the cross validation for that 
feature and goes onto the next feature.