{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "[5.2] Create in src/models/pytorch.py a class called PytorchRegression that inherits from nn.Module with:\n",
    "\n",
    "* `num_features` as input parameter \n",
    "* attributes:\n",
    " - `layer_1`: fully-connected layer with 128 neurons\n",
    " - `layer_out`: fully-connected layer with 1 neurons\n",
    "* methods:\n",
    " - `forward()` with `inputs` as input parameter, perform ReLU and DropOut on the fully-connected layer followed by the output layer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class PytorchRegression(nn.Module):\n",
    "    def __init__(self, num_features):\n",
    "        super(PytorchRegression, self).__init__()\n",
    "        \n",
    "        self.layer_1 = nn.Linear(num_features, 128)\n",
    "        self.layer_out = nn.Linear(128, 1)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = F.dropout(F.relu(self.layer_1(x)))\n",
    "        x = self.layer_out(x)\n",
    "        return (x)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
