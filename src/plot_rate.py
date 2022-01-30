from copygec.models.optimizer import NoamOpt
import matplotlib.pyplot as plt
import numpy as np

# Three settings of the lrate hyperparameters.
opts = [NoamOpt(512, .25, 200, None), 
        NoamOpt(512, .25, 200, None),
        NoamOpt(256, .25, 200, None)]
plt.plot(np.arange(1, 5000), [[opt.rate(i) for opt in opts] for i in range(1, 5000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
plt.show()