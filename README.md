The repository for the public facing site that will host the data visualization handbook. The jupyter-book is in the `handbook` directory, and can be run by navigating to it in your console and executing `jupyter-book build .` then following the instructions to paste a link into your browser. 

This requires Python to be on the PATH, as well as all of the packages in `requirements.txt` to be installed, which can be done by running `pip install -r requirements.txt` while in the `handbook` directory. The [similaritymeasures](https://pypi.org/project/similaritymeasures/) library is very important. You must also have access to the Google Cloud project; instructions forthcoming. 

The `handbook/images` directory contains sample plots generated from the notebooks and have unique file names. They show a range of methods and parameters that can be used to create the graphs, and hopefully show how they change the runtime and output. 
