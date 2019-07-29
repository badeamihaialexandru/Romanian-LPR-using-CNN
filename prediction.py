import os
import numpy as np
import segmentation
import matplotlib.pyplot as plt
from sklearn.externals import joblib
print("printing chars: \n")
for each in segmentation.characters:
    fig, (ax5) = plt.subplots(1, 1)
    ax5.imshow(each, cmap="gray")
    plt.show()
print("done printing! \n")

"""
# load the model
current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')
model = joblib.load(model_dir)

classification_result = []
for each_character in segmentation.characters:
    # converts it to a 1D array
    each_character = each_character.reshape(1, -1);
    result = model.predict(each_character)
    classification_result.append(result)


print(classification_result)

plate_string = ''
for eachPredict in classification_result:
    plate_string += eachPredict[0]
print()
print(plate_string)

# it's possible the characters are wrongly arranged
# since that's a possibility, the column_list will be
# used to sort the letters in the right order

column_list_copy = segmentation.column_list[:]
print("segm col : ",segmentation.column_list)
segmentation.column_list.sort()

rightplate_string = ''
for each in segmentation.column_list:
    rightplate_string += plate_string[column_list_copy.index(each)]

print(rightplate_string)

"""