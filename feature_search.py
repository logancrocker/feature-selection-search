import numpy as np 
import random
import sys
import timeit

#cross validation function
def cross_validation(labels, features, feature_set, optimize, best_acc):
    #grab only the features we are currently checking
    current_features = features[:, feature_set]
    new_labels = []
    count = 0
    if (optimize):
        max_errors_allowed = 200 - best_acc
        errors = 0
    #print(current_features)
    #loop through all instances 
    for x in range(200):
        current_test_instance = current_features[x]
        best_distance = 1000000
        nearest_class = 0
        #compare to all instances
        for y in range(200):
            compare_to_this = current_features[y]
            distance = np.linalg.norm(current_test_instance - compare_to_this)
            #if distance is smaller than current best and not zero, update it
            if (distance != 0 and distance < best_distance):
                best_distance = distance
                nearest_class = labels[y]
        if (nearest_class == labels[x]):
            count += 1
        else:
            if (optimize):
                errors += 1
                if (errors > max_errors_allowed):
                    return "Skipped!"

    return count

def forward_search(data):
    #get class labels
    labels = data[:, 0]
    #get features
    features = data[:, 1:]
    numInstances, numFeatures = features.shape
    feature_set = []
    best_feature_set = []
    best_percentage = 0
    #walk through every level of the search tree
    for level in range(numFeatures):
        print("On level " + str(level + 1))
        best_acc = 0
        feature_to_add_at_this_level = []
        #consider each feature
        for feature in range(numFeatures):
            #only check a feature if it not already in the set 
            if (feature not in feature_set):
                acc = cross_validation(labels, features, feature_set + [feature], False, best_acc)
                percentage = (acc / 200.0) * 100
                sys.stdout.write("\tConsidering feature " + str(feature + 1) + " -> " + str(percentage) + "%\n")
                sys.stdout.flush()
                if (acc > best_acc):
                    best_acc = acc
                    feature_to_add_at_this_level = feature
                if (percentage > best_percentage):
                    best_percentage = percentage
                    sys.stdout.flush()
                    best_feature_set = feature_set + [feature]

        #add the feature to the set
        feature_set.append(feature_to_add_at_this_level)
        print("\t\tOn level " + str(level + 1) + " I added feature " + str(feature_to_add_at_this_level + 1))
    output_set = [val+1 for val in best_feature_set]
    print("Best accuracy was " + str(best_percentage) + "% with feature set " + str(output_set))


def backward_search(data, size):
    #get class labels
    labels = data[:, 0]
    #get features
    features = data[:, 1:]
    numInstances, numFeatures = features.shape
    if (size == 1):
        feature_set = [0,1,2,3,4,5,6,7,8,9]
    elif (size == 2):
        feature_set = [ x for x in range(100) ]
    best_feature_set = []
    best_percentage = 0
    #walk through every level of the search tree
    for level in range(numFeatures):
        #print(feature_set)
        print("On level " + str(level + 1))
        best_acc = 0
        feature_to_remove_at_this_level = []
        #consider each feature
        for feature in range(numFeatures):
            #consider removing all features still in the set 
            if (feature in feature_set):
                #make new list that doesnt contain current feature
                new_list = [ x for x in feature_set if x != feature ]
                #classify
                acc = cross_validation(labels, features, new_list, False, best_acc)
                percentage = (acc / 200.0) * 100
                sys.stdout.write("\tConsidering removing feature " + str(feature + 1) + " -> " + str(percentage) + "%\n")
                sys.stdout.flush()
                if (acc > best_acc):
                    best_acc = acc
                    feature_to_remove_at_this_level = feature
                elif (acc == 0):
                    feature_to_remove_at_this_level = feature
                if (percentage > best_percentage):
                    best_percentage = percentage
                    sys.stdout.flush()
                    best_feature_set = new_list

        #remove the feature from the set
        #print(feature_to_remove_at_this_level)
        feature_set.remove(feature_to_remove_at_this_level)
        print("\t\tOn level " + str(level + 1) + " I removed feature " + str(feature_to_remove_at_this_level + 1))
    output_set = [val+1 for val in best_feature_set]
    print("Best accuracy was " + str(best_percentage) + "% with feature set " + str(output_set))


def forward_search_faster(data):
    #get class labels
    labels = data[:, 0]
    #get features
    features = data[:, 1:]
    numInstances, numFeatures = features.shape
    feature_set = []
    best_feature_set = []
    best_percentage = 0
    #walk through every level of the search tree
    for level in range(numFeatures):
        print("On level " + str(level + 1))
        best_acc = 0
        feature_to_add_at_this_level = []
        #consider each feature
        for feature in range(numFeatures):
            #only check a feature if it not already in the set 
            if (feature not in feature_set):
                acc = cross_validation(labels, features, feature_set + [feature], True, best_acc)
                if (acc != "Skipped!"):
                    percentage = (acc / 200.0) * 100
                    sys.stdout.write("\tConsidering feature " + str(feature + 1) + " -> " + str(percentage) + "%\n")
                    sys.stdout.flush()
                else:
                    sys.stdout.write("\tSkipped feature " + str(feature + 1) + " due to too many errors\n")
                    sys.stdout.flush()
                if (acc != "Skipped!" and acc > best_acc):
                    best_acc = acc
                    feature_to_add_at_this_level = feature
                if (percentage > best_percentage):
                    best_percentage = percentage
                    sys.stdout.flush()
                    best_feature_set = feature_set + [feature]

        #add the feature to the set
        feature_set.append(feature_to_add_at_this_level)
        print("\t\tOn level " + str(level + 1) + " I added feature " + str(feature_to_add_at_this_level + 1))
    output_set = [val+1 for val in best_feature_set]
    print("Best accuracy was " + str(best_percentage) + "% with feature set " + str(output_set))

print("Select which dataset you would like to use.")
print("\t1. Small dataset")
print("\t2. Large dataset")

choice = None
while (choice != 1) and (choice != 2): 
  try:
      choice = int(input("Choice: "))
  except:
      print("ERROR: Make a choice or enter 0 to quit.")
      continue
  if (choice == 0):
      print("Goodbye.")
      sys.exit()
  elif (choice == 1):
      size = choice
      data = np.loadtxt("small_data.txt")
  elif (choice == 2):
      size = choice
      data = np.loadtxt("large_data.txt")
  elif (choice == 'quit'):
      sys.quit()
  else:
      print("ERROR: Invalid choice !!!")

print("Select which algorithm you want to use.")
print("\t1. Forward selection")
print("\t2. Backward deletion")
print("\t3. Optimized forward selection")

choice = None
while (choice != 1) and (choice != 2) and (choice != 3): 
  try:
      choice = int(input("Choice: "))
  except:
      print("ERROR: Make a choice or enter 0 to quit.")
      continue
  if (choice == 0):
      print("Goodbye.")
      sys.exit()
  elif (choice == 1):
      tic = timeit.default_timer()
      forward_search(data)
      toc = timeit.default_timer()
      print("Forward search completed after " + str(toc - tic) + " seconds.")
  elif (choice == 2):
      tic = timeit.default_timer()
      backward_search(data, size)
      toc = timeit.default_timer()
      print("Backwards search completed after " + str(toc - tic) +" seconds.")
  elif (choice == 3):
      tic = timeit.default_timer()
      forward_search_faster(data)
      toc = timeit.default_timer()
      print("Optimized forward search completed after " + str(toc - tic) + " seconds.")
