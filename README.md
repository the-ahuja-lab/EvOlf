# EvOlf
<!--- BADGES --->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Prompt-Optimization.ipynb)
<!--- Update colab path --->
<!--- ADD Paper when published --->

EvOlf is a cutting-edge deep-learning framework designed to predict ligand-GPCR interactions, integrating odorant and non-odorant GPCRs across 24 species. Using advanced modeling techniques, it enables accurate predictions and GPCR deorphanization. <br><br>
Want to see it in action? We’ve provided a Google Colab notebook so you can easily run predictions on your own data! 🚀
👉 EvOlf Colab Notebook
<!--- Update colab path --->

## Get Started
Ready to dive in? First, clone this repository to your chosen location:
```bash
git clone https://github.com/the-ahuja-lab/EvOlf.git
```
Here's a key thing to remember: Your working directory is the folder where you cloned `EvOlf`, not `EvOlf` itself! For example, if you clone EvOlf inside a folder called `MyProjects`, then `MyProjects` is your working directory.

An example of Input Data
| ID | Ligand_ID | SMILES | Receptor_ID| Sequence |
|:-----|:---------|:-----------|:---------|:-----------|
|LR1  |L1  |OCC/C=C\CC  |R1  | MTVENYSTATQFVLAGLTQQAEIQLPLFLLFLGIYLVTVVGNLGMVLLIAVSPLLHTPMYYFLSSLSFVDFCYSSVITPKMLLNFLGKNTILYSECMVQLFFFVVFVVAEGYLLTAMAYDRYVAICSPLLYNVIMSSWVCSPLVLAAFFLGFLSALAHTSAMMKLSFCKSHIINHYFCDVLPLLNLSCSNTYLNELLLFIIAGFNTLVPTLAVAISYAFIFYSILHIRSSEGRSKAFGTCSSHLMAVGIFFGSITFMYFKPPSSNSLDQEKVSSVFYTTVIPMLNPLIYSLRNKDVKKALRKVLVGK|
|LR2  |L1  |OCC/C=C\CC  |R2  |  |
|LR3  |L2  |CCCC(=O)O  |R3  |  |
|LR4  |L3  |CCCCCCCC=O  |R3  |  |
|LR5  |L4  |CC(=O)C(=O)C  |R4  |  |
