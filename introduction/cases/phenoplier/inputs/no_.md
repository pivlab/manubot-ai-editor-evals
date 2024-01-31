Genes work together in context-specific networks to carry out different functions1,2. Variations in these genes can change their functional role and, at a higher level, affect disease-relevant biological processes3. In this context, determining how genes influence complex traits requires mechanistically understanding expression regulation across different cell types4–6, which in turn should lead to improved treatments7,8. Previous studies have described different regulatory DNA elements5,9–12 including genetic effects on gene expression across different tissues4. Integrating functional genomics data and GWAS data13,13–15 has improved the identification of these transcriptional mechanisms that, when dysregulated, commonly result in tissue- and cell lineage-specific pathology16–18.

Given the availability of gene expression data across several tissues4,19–21, an effective approach to identify these biological processes is the transcription-wide association study (TWAS), which integrates expression quantitative trait loci (eQTLs) data to provide a mechanistic interpretation for GWAS findings. TWAS relies on testing whether perturbations in gene regulatory mechanisms mediate the association between genetic variants and human diseases22–25, and these approaches have been highly successful not only in understanding disease etiology at the transcriptome level26–28 but also in disease-risk prediction (polygenic scores)29 and drug repurposing30 tasks. However, TWAS works at the individual gene level, which does not capture more complex interactions at the network level.

These gene-gene interactions play a crucial role in current theories of the architecture of complex traits, such as the omnigenic model31, which suggests that methods need to incorporate this complexity to disentangle disease-relevant mechanisms. Widespread gene pleiotropy, for instance, reveals the highly interconnected nature of transcriptional networks32,33, where potentially all genes expressed in disease-relevant cell types have a non-zero effect on the trait31,34. One way to learn these gene-gene interactions is using the concept of gene module: a group of genes with similar expression profiles across different conditions2,35,36. In this context, several unsupervised approaches have been proposed to infer these gene-gene connections by extracting gene modules from co-expression patterns37–39. Matrix factorization techniques like independent or principal component analysis (ICA/PCA) have shown superior performance in this task40 since they capture local expression effects from a subset of samples and can handle modules overlap effectively. Therefore, integrating genetic studies with gene modules extracted using unsupervised learning could further improve our understanding of disease origin36 and progression41.

Here we propose PhenoPLIER, an omnigenic approach that provides a gene module perspective to genetic studies. The flexibility of our method allows integrating different data modalities into the same representation for a joint analysis. We show that this module perspective can infer how groups of functionally-related genes influence complex traits, detect shared and distinct transcriptomic properties among traits, and predict how pharmacological perturbations affect genes’ activity to exert their effects. PhenoPLIER maps gene-trait associations and drug-induced transcriptional responses into a common latent representation. For this, we integrate thousands of gene-trait associations (using TWAS from PhenomeXcan42) and transcriptional profiles of drugs (from LINCS L100043) into a low-dimensional space learned from public gene expression data on tens of thousands of RNA-seq samples (recount219,44). We use a latent representation defined by a matrix factorization approach44,45 that extracts gene modules with certain sparsity constraints and preferences for those that align with prior knowledge (pathways). When mapping gene-trait associations to this reduced expression space, we observe that diseases are significantly associated with gene modules expressed in relevant cell types: such as hypothyroidism with T cells, corneal endothelial cells with keratometry measurements, hematological assays on specific blood cell types, plasma lipids with adipose tissue, and neuropsychiatric disorders with different brain cell types. Moreover, since PhenoPLIER can use models derived from large and heterogeneous RNA-seq datasets, we can also identify modules associated with cell types under specific stimuli or disease states. We observe that significant module-trait associations in PhenomeXcan (our discovery cohort) replicated in the Electronic Medical Records and Genomics (eMERGE) network phase III27,46 (our replication cohort). Furthermore, we perform a CRISPR screen to analyze lipid regulation in HepG2 cells. We observe more robust trait associations with modules than with individual genes, even when single genes known to be involved in lipid metabolism did not reach genome-wide significance. Compared to a single-gene approach, our module-based method also better predicts FDA-approved drug-disease links by capturing tissue-specific pathophysiological mechanisms linked with the mechanism of action of drugs (e.g., niacin with cardiovascular traits via a known immune mechanism). This improved drug-disease prediction suggests that modules may provide a better means to examine drug-disease relationships than individual genes. Finally, exploring the phenotype-module space reveals stable trait clusters associated with relevant tissues, including a complex branch involving lipids with cardiovascular, autoimmune, and neuropsychiatric disorders. In summary, instead of considering single genes associated with different complex traits, PhenoPLIER incorporates groups of genes that act together to carry out different functions in specific cell types. This approach improves robustness in detecting and interpreting genetic associations, and here we show how it can prioritize alternative and potentially more promising candidate targets even when known single gene associations are not detected. The approach represents a conceptual shift in the interpretation of genetic studies. It has the potential to extract mechanistic insight from statistical associations to enhance the understanding of complex diseases and their therapeutic modalities.