S-PrediXcan [@doi:10.1038/s41467-018-03621-1] is the summary version of PrediXcan [@doi:10.1038/ng.3367]. PrediXcan models the trait as a linear function of the gene's expression on a single tissue using the univariate model

$$
\mathbf{y} = \mathbf{t}_l \gamma_l + \bm{\epsilon}_l,
$$ {#eq:predixcan}

where $\hat{\gamma}_l$ is the estimated effect size or regression coefficient, and $\bm{\epsilon}_l$ are the error terms with variance $\sigma_{\epsilon}^{2}$. The significance of the association is assessed by computing the $z$-score $\hat{z}_{l}=\hat{\gamma}_l / \mathrm{se}(\hat{\gamma}_l)$ for a gene's tissue model $l$. PrediXcan needs individual-level data to fit this model, whereas S-PrediXcan approximates PrediXcan $z$-scores using only GWAS summary statistics with the expression

$$
\hat{z}_{l} \approx \sum_{a \in model_{l}} w_a^l \frac{\hat{\sigma}_a}{\hat{\sigma}_l} \frac{\hat{\beta}_a}{\mathrm{se}(\hat{\beta}_a)},
$$ {#eq:spredixcan}

where $\hat{\sigma}_a$ is the variance of SNP $a$, $\hat{\sigma}_l$ is the variance of the predicted expression of a gene in tissue $l$, and $\hat{\beta}_a$ is the estimated effect size of SNP $a$ from the GWAS. In these TWAS methods, the genotype variances and covariances are always estimated using the Genotype-Tissue Expression project (GTEx v8) [@doi:10.1126/science.aaz1776] as the reference panel. Since S-PrediXcan provides tissue-specific direction of effects (for instance, whether a higher or lower predicted expression of a gene confers more or less disease risk), we used the $z$-scores in our drug repurposing approach (described below).