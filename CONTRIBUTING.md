# Contributing to epidemic-intelligence
`epidemic-intelligence` is in a nascient stage and will require many tweaks to ensure optimal functionality going forward, not to mention the large additions to said functionality. This document contains instructions, primarily targeted towards NetSI contributors, for updating the package's Python Package Index entry and documentation. 

## Updating the package
### Function conventions
`epidemic-intelligence` is organized into three modules: boxplots, importation_plots, and processing. All of the main user-facing functions are exposed via the init, so it is not necessary for the user to specify the module when using a function. Because of this, it is recommended that all functions have distinct names, even if they are in seperate modules. 

The `helper.py` file within the helper directory contains several functions, including the geographic filter, than should be used when constructing new visualizations. 

### Templates
Each file in the templates directory uses the plotly.io module to create a [plotly template](https://plotly.com/python/templates/).

Further updates to the library are required to make more robust use of different templates. 

### Building and pushing the package
#### 1. Last checks
Before the package is pushed, the version number must be changed in `pyproject.toml`. The full release of `epidemic-intelligence` uses semantic versioning, of which the full explanation can be found [here](https://semver.org/).

Any changes to the full release of `epidemic-intelligence` should be reflected in the documentation. See below for instructions. 

Ensure `twine` is installed:
```
    pip install twine
```

#### 2. Build dist files
Run the following command from the `/library` directory of the repository:
```
    python -m build
```

If successful, the `/library/dist` folder will populate with a .whl and a .tar.gz file corresponding to the version number. 

#### 3. Push to server
You must have a PyPI token to push either the test or full release of `epidemic-intelligence`. You will need a different token for each one. 

It is recommended that you use twine to push the package to the server. 

To push to the test PyPI server: 
```
    twine upload --repository testpypi dist/* --username __token__ --password paste_your_token_here
```
To push to the full PyPI server:
```
    twine upload dist/* --username __token__ --password paste_your_token_here
```

Note that all versions with a .tar.gz and a .whl file in the `\library\dist` folder will be uploaded. To prevent this, it is recommended that you remove the files corresponding to older versions from the folder before pushing to either server. 

## Updating documentation
The `epidemic-intelligence` documentation is a Jupyter Book published [here](url). It includes full docstrings and examples for each user-facing function in the package. Any changes to the functionality of `epidemic-intelligence` should include corresponding changes to the documentation. 

### Documentation structure
`\hanbook\content` contains the `.ipynb` files that form the pages of the Jupyter Book. `\handbook\images` contains the static image files that will be displayed in the Jupyter Book. 

All user-facing functions should include a full docstring and at least one complete example, if applicable. Reference current documentation structure to ensure continuity. 

#### BigQuery credentials
Place your BigQuery credentials in the same directory as this repository, and ensuring that the file is named `credentials.json`. From a file within the `\handbook\content` directory, the relative path to these credentials will be `..\..\..\credentials.json`.  

### Pushing documentation
#### 1. Last checks
Ensure that the table of contents, found at `\handbook\_toc.yaml`, is up to date. Documentation for structuring the table of contents can be found [here](https://jupyterbook.org/en/stable/structure/toc.html). 

#### 2. Building html
Before building new HTML for the documentation, it is recommended that you delete the existing `\handbook\_build` folder, if it exists. 

Run the following command from the `\handbook` directory to create the `\handbook\_build` folder. This command will run all notebooks referenced within the table of contents:
```
    python -m build
```

#### 3. CHECK YOUR WORK!
Before updating the live site, please check that the local build works! The output of the previous command will include instructions for viewing the local build in your browser. 

#### 4. Update gh-pages
Once you are ready to push to the live site, run the following command. This will overwrite the gh-pages branch of this repository, which is the basis for the live site:
```
    ghp-import -n -p -f _build/html
```
