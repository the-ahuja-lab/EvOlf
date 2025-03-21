# EvOlf

EvOlf is a cutting-edge deep-learning framework designed to predict ligand-GPCR interactions, integrating odorant and non-odorant GPCRs across 24 species. Using advanced modeling techniques, it enables accurate predictions and GPCR deorphanization. 
Want to see it in action? We’ve provided a Google Colab notebook so you can easily run predictions on your own data! 🚀

Get Started
Ready to dive in? First, clone this repository to your local machine using:
```
git clone https://github.com/YourUsername/EvOlf.git
cd EvOlf
```

An example of Input Data
| ID | Ligand_ID | SMILES | Receptor_ID| Sequence |
|:-----|:---------|:-----------|:---------|:-----------|
|LR1  |L1  |OCC/C=C\CC  |R1  | MTVENYSTATQFVLAGLTQQAEIQLPLFLLFLGIYLVTVVGNLGMVLLIAVSPLLHTPMYYFLSSLSFVDFCYSSVITPKMLLNFLGKNTILYSECMVQLFFFVVFVVAEGYLLTAMAYDRYVAICSPLLYNVIMSSWVCSPLVLAAFFLGFLSALAHTSAMMKLSFCKSHIINHYFCDVLPLLNLSCSNTYLNELLLFIIAGFNTLVPTLAVAISYAFIFYSILHIRSSEGRSKAFGTCSSHLMAVGIFFGSITFMYFKPPSSNSLDQEKVSSVFYTTVIPMLNPLIYSLRNKDVKKALRKVLVGK|
|LR2  |L1  |OCC/C=C\CC  |R2  |  |
|LR3  |L2  |CCCC(=O)O  |R3  |  |
|LR4  |L3  |CCCCCCCC=O  |R3  |  |
|LR5  |L4  |CC(=O)C(=O)C  |R4  |  |
