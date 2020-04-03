# Explanation of the structure of the results folder
For each filter the top4_avg (top 4 of the images of the tinyimagenet dataset that lead to the highest average output) and top4_max are stored in the json (they just reference the image in the tinyimagenet/exsmallimagenet folder.
## avg_grads
The gradients of the top4_avg of each filter with respect to that filter.

## max_grads
Equivalent of avg_grads for top4_avg with respect to top4_max

## cropped
for each image in the top4_max (the image of the tinyimagenet that managed to enhance the highest scalar score for the given filter), there is a region in the original image that lead to that pixel with highest scalar value. the cropped folder keep this cropped version of the top4_max images.

## cropped_grad
this same cropping is applied to the gradients of top4_max

## max_activ
the artificial image obtained by gradient descent which gives the highest average output score for a given filter.

To summary we have:
- 1: top4_avg
- 2: top4_max
- 3: avg_grads (deconvolution of top4_avg)
- 4: max_grads (deconvolution of top4_max)
- 5: cropped (cropped version of top4_max, slice of the original image resulting in the value which allowed the image to belong to top4_max
- 6: cropped_grad (same cropping applied to max_grads)
- 7: max_activ
