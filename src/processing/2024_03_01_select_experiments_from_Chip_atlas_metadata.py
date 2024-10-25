# fn='/notebook/human_lines_cistrome/chip-atlas/experimentList_human_TFs_HG38.txt'
fn = "/data1/datasets_1/human_cistrome/chip-atlas/2024_04_10_ChIP-atlas_experimentList.tab"
fn_out = "/data1/datasets_1/human_cistrome/chip-atlas/2024_04_10_ChIP-atlas_experimentList_parsed.tab"
fn_out_rejected = "/data1/datasets_1/human_cistrome/chip-atlas/2024_04_10_ChIP-atlas_experimentList_parsed_rejected.tab"
# fn_out='/notebook/human_lines_cistrome/chip-atlas/experimentList_human_TFs_HG38_parsed.txt'
# fn_out_rejected = '/notebook/human_lines_cistrome/chip-atlas/experimentList_human_TFs_HG38_parsed_rejected.txt'
f = open(fn)
fo = open(fn_out, "w")
fo_rejected = open(fn_out_rejected, "w")
fo.write(
    "IDENTIFIER\tTF\tcell_line\tdescription\ttreatment\ttreatment_duration\tsource\tn_mapped\tpct_mapped\tpct_dup\tn_peaks\n"
)
fo_rejected.write(
    "IDENTIFIER\tTF\tcell_line\tdescription\ttreatment\ttreatment_duration\tsource\tn_mapped\tpct_mapped\tpct_dup\tn_peaks\n"
)

for line in f:
    keep = True
    line = line.replace("'", "")
    line = line.replace(";", " ")
    line = line.replace("#", "")
    a = line.rstrip("\r\n").split("\t")
    id = a[0]
    tf = a[3]
    cell_line = a[5]
    try:
        n_mapped, pct_mapped, pct_dup, n_peaks = a[7].split(",")
    except:
        continue
    description = a[8]
    treatment = "NA"
    treatment_duration = "NA"
    source = "NA"
    if len(a) > 9:
        for slug in a[9 : len(a)]:
            if slug.count("=") == 1:
                key, val = slug.split("=")
            else:
                x = slug.split("=")
                key = x[0]
                val = "=".join(x[1 : len(x)])

            if key == "source_name":
                source = val

            if key == "treatment":
                treatment = val

            if key == "treatment_duration":
                treatment_duration = val

            if key == "stimulus":
                if val.count("inhibitor") > 0:
                    keep = False

            if key == "genotype":
                if val.count("p53-R") > 0:
                    keep = False

            if key == "Title":
                if val.count("Sh1") > 0:
                    keep = False

            if key == "agent":
                if val.count("ATRA") > 0 or val.count("zinc") > 0:
                    keep = False

            if key == "transfection":
                if val.count("BAC") > 0:
                    keep = False

    organism = a[1]
    if organism != "hg38":
        keep = False

    if keep:
        if (
            tf == "ATAC-Seq"
            or tf == "GFP"
            or tf == "Epitope tags"
            or tf == "MethylCap"
            or tf.count("Cyclobutane") > 0
            or tf.count("acetyllysine") > 0
            or tf == "Hepatitis B Virus X antigen"
            or tf == "5-hmC"
            or tf == "5-mC"
            or tf == "8-Hydroxydeoxyguanosine"
            or tf == "AAV Rep68"
            or tf == "Adenine N6-methylation"
            or tf == "Bisulfite-Seq"
            or tf.count("ChromHMM") > 0
            or tf == "ClinVar"
            or tf == "CpG Islands"
            or tf.count("rotony") > 0
            or tf == "DNase-Seq"
            or tf == "ENCODE Blacklist"
            or tf.count("genes") > 0
            or tf.count("eQTL") > 0
            or tf.count("FANTOM5") > 0
            or tf == "GWAS Catalog"
            or tf.count("Hi-C") > 0
            or tf == "Input control"
            or tf == "JASPAR TF motif"
            or tf.count("H3K") > 0
            or tf == "MGI Phenotype"
            or tf == "Orphanet"
            or tf.count("PhastCons") > 0
            or tf == "RepeatMasker"
            or tf.count("polymerase") > 0
            or tf.count("RNA-seq") > 0
            or tf == "Unclassified"
        ):
            keep = False

    tf = tf.upper()

    if keep:
        if (
            cell_line.count("Keratinocytes") > 0
            or cell_line.count("Large pulmonary artery endothelial cells") > 0
            or cell_line.count("Adipo") > 0
            or cell_line.count("Acute ") > 0
            or cell_line.count("Adenoid") > 0
            or cell_line.count("ALL xenograft") > 0
            or cell_line.count("Anterior temporal cortex") > 0
            or cell_line.count("Aortic ") > 0
            or cell_line.count("B cells") > 0
            or cell_line.count("B Lymphoblastoid cell line") > 0
            or cell_line.count("B-CLL cells") > 0
            or cell_line.count("Bone marrow ") > 0
            or cell_line.count("Breast ") > 0
            or cell_line.count("Bronchial ") > 0
            or cell_line.count("Burkitt Lymphoma") > 0
            or cell_line.count("Caput") > 0
            or cell_line.count("Carcinoma, ") > 0
            or cell_line.count("Cardiac ") > 0
            or cell_line.count("Chondrocytes") > 0
            or cell_line.count("Cord blood") > 0
            or cell_line.count("leukemic") > 0
            or cell_line.count("CD34") > 0
            or cell_line.count("CD4+ T cells") > 0
            or cell_line.count("Chronic myeloid leukemia") > 0
            or cell_line.count("Colon cancer") > 0
            or cell_line.count("Colonic organoids") > 0
            or cell_line.count("Colorectal ") > 0
            or cell_line.count("Coronary ") > 0
            or cell_line.count("Dendritic Cells") > 0
            or cell_line.count("Dermal fibroblast") > 0
            or cell_line.count("Differentiated keratinocytes") > 0
            or cell_line.count("Diffuse intrinsic pontine glioma") > 0
            or cell_line.count("Endometrial ") > 0
            or cell_line.count("Endometrium") > 0
            or cell_line.count("Epidermal ") > 0
            or cell_line.count("Epididymis") > 0
            or cell_line.count("Endothelial") > 0
            or cell_line.count("Erythr") > 0
            or cell_line.count("ES cells") > 0
            or cell_line.count("Fallopian Tubes") > 0
            or cell_line.count("Fetal ") > 0
            or cell_line.count("Fibro") > 0
            or cell_line.count("Fore") > 0
            or cell_line.count("Gastrointestinal") > 0
            or cell_line.count("Glioblastoma") > 0
            or cell_line.count("Glioma") > 0
            or cell_line.count("Head and neck ") > 0
            or cell_line.count("Heart") > 0
            or cell_line.count("Hematopoietic") > 0
            or cell_line.count("hESC ") > 0
            or cell_line.count("Hepatoc") > 0
            or cell_line.count("Hippocampus") > 0
            or cell_line.count("iPS") > 0
            or cell_line.count("Kidney") > 0
            or cell_line.count("Epithelium, Corneal") > 0
            or cell_line.count("Large pulmonary artery endothelial cells") > 0
            or cell_line.count("Leiomyoma") > 0
            or cell_line.count("Leukemia, Myeloid") > 0
            or cell_line.count("Localized") > 0
            or cell_line.count("Liver") > 0
            or cell_line.count("LTED cells") > 0
            or cell_line.count("Lung") > 0
            or cell_line.count("Lung cancer cell line") > 0
            or cell_line.count("Lung fibroblasts") > 0
            or cell_line.count("ymphoblast") > 0
            or cell_line.count("Lymphoma, B-Cell") > 0
            or cell_line.count("Macrophages") > 0
            or cell_line.count("Mammary ") > 0
            or cell_line.count("Mast Cells") > 0
            or cell_line.count("Melanocytes") > 0
            or cell_line.count("Melanoma") > 0
            or cell_line.count("Memory T cells") > 0
            or cell_line.count("Mesenchymal stem cells") > 0
            or cell_line.count("Monocytes") > 0
            or cell_line.count("Multiple Myeloma") > 0
            or cell_line.count("Muscle") > 0
            or cell_line.count("Myoblasts") > 0
            or cell_line.count("Myofibroblasts") > 0
            or cell_line.count("Myocytes") > 0
            or cell_line.count("Myometrium") > 0
            or cell_line.count("Myotube") > 0
            or cell_line.count("Neural progenitor \cells") > 0
            or cell_line.count("Neuro") > 0
            or cell_line.count("Neural ") > 0
            or cell_line.count("Nuclear ") > 0
            or cell_line.count("Neutrophils") > 0
            or cell_line.count("Osteoblasts") > 0
            or cell_line.count("Ovarian granulosa cells") > 0
            or cell_line.count("Pancre") > 0
            or cell_line.count("PBMC") > 0
            or cell_line.count("PBDEFetal") > 0
            or cell_line.count("Peripheral ") > 0
            or cell_line.count("Pharynx cancer cells") > 0
            or cell_line.count("Placenta") > 0
            or cell_line.count("Plasma Cells") > 0
            or cell_line.count("Pre-adipocytes") > 0
            or cell_line.count("Pre-B cell leukemia") > 0
            or cell_line.count("Pre-leukemia stem cells") > 0
            or cell_line.count("Prefrontal Cortex") > 0
            or cell_line.count("Primary") > 0
            or cell_line.count("Prostat") > 0
            or cell_line.count("Renal ") > 0
            or cell_line.count("Retina") > 0
            or cell_line.count("Rhabdoid Tumor") > 0
            or cell_line.count("Rhabdomyosarcoma") > 0
            or cell_line.count("Skeletal") > 0
            or cell_line.count("Small cell ovary carcinoma") > 0
            or cell_line.count("Spleen") > 0
            or cell_line.count("T cells") > 0
            or cell_line.count("Testis") > 0
            or cell_line.count("Th1 Cells") > 0
            or cell_line.count("Th2 Cells") > 0
            or cell_line.count("Th17 Cells") > 0
            or cell_line.count("Thymus") > 0
            or cell_line.count("Tracheal epithelial cells") > 0
            or cell_line.count("Treg") > 0
            or cell_line.count("Tumor") > 0
            or cell_line.count("Tumour") > 0
            or cell_line.count("Unclassified") > 0
            or cell_line.count("Uterin leiomyoma") > 0
            or cell_line.count("harton Jelly") > 0
            or cell_line.count("omental") > 0
            or cell_line.count("ibial nerve") > 0
            or cell_line.count("olon") > 0
            or cell_line.count("transverse") > 0
            or cell_line.count("xcitatory neuron") > 0
            or cell_line.count("sophagus squamous epithelium") > 0
            or cell_line.count("astrocnemius medialis") > 0
            or cell_line.count("ody of pancreas") > 0
            or cell_line.count("rontal cortex") > 0
            or cell_line.count("eyers patches") > 0
            or cell_line.count("strocytes") > 0
            or cell_line.count("ight atrium auricular region") > 0
            or cell_line.count("sophagus muscularis mucosa") > 0
            or cell_line.count("Mesenchymal stromal cells") > 0
            or cell_line.count("olonic mucosa") > 0
            or cell_line.count("rain") > 0
            or cell_line.count("icroglia") > 0
            or cell_line.count("terus") > 0
            or cell_line.count("ymph nodes") > 0
            or cell_line.count("orta") > 0
            or cell_line.count("thoracic") > 0
            or cell_line.count("ubcutaneous adipose tissue") > 0
            or cell_line.count("hyroid gland") > 0
            or cell_line.count("ibial arteries") > 0
            or cell_line.count("astroesophageal sphincter") > 0
            or cell_line.count("drenal glands") > 0
            or cell_line.count("agina") > 0
            or cell_line.count("atural killer cells") > 0
            or cell_line.count("vary") > 0
            or cell_line.count("Adrenal glands") > 0
            or cell_line.count("Alveolar soft part sarcoma") > 0
            or cell_line.count("Aneurysm smooth muscle cells") > 0
            or cell_line.count("B-all") > 0
            or cell_line.count("B-cell (cd19+)") > 0
            or cell_line.count("Bipolar spindle neurons") > 0
            or cell_line.count("Bladder cancer") > 0
            or cell_line.count("Bone marrow cells") > 0
            or cell_line.count("Cardiomyocytes") > 0
            or cell_line.count("Cd20+") > 0
            or cell_line.count("Cd36+") > 0
            or cell_line.count("Cd4+") > 0
            or cell_line.count("Cerebellum") > 0
            or cell_line.count("Cytotrophoblast") > 0
            or cell_line.count("Embryonic kidney") > 0
            or cell_line.count("Epithelial cells") > 0
            or cell_line.count("Esophageal squamous carcinoma cell line") > 0
            or cell_line.count("Glioblstoma stem cell") > 0
            or cell_line.count("Gonadal somatic cells") > 0
            or cell_line.count("Granulocytes") > 0
            or cell_line.count("Hodgkins lymphoma cell line") > 0
            or cell_line.count("Lateral tempor \al lobe") > 0
            or cell_line.count("Leg skin") > 0
            or cell_line.count("Limbal stem cells") > 0
            or cell_line.count("Lymph nodes") > 0
            or cell_line.count("Mesenchymal stromal cells") > 0
            or cell_line.count("Mesothelial cells") > 0
            or cell_line.count("Mycosis fungoides") > 0
            or cell_line.count("Myelomonocytic leukemia") > 0
            or cell_line.count("Na") > 0
            or cell_line.count("Natural killer t-cells") > 0
            or cell_line.count("Ntera2-derived neural stem cells") > 0
            or cell_line.count("Omental fat pad") > 0
            or cell_line.count("or \al squamous cell carcinoma") > 0
            or cell_line.count("Osteocytes") > 0
            or cell_line.count("Osteosarcoma") > 0
            or cell_line.count("Pericytes") > 0
            or cell_line.count("Peyers patches") > 0
            or cell_line.count("Plasmablasts") > 0
            or cell_line.count("Podocytes") > 0
            or cell_line.count("Pre-b all ") > 0
            or cell_line.count("Preadipocytes") > 0
            or cell_line.count("Primor \dial germ cells") > 0
            or cell_line.count("Sarcoma, clear cell") > 0
            or cell_line.count("Skin fibroblasts") > 0
            or cell_line.count("Small cell lung cancer") > 0
            or cell_line.count("Smooth muscle cells") > 0
            or cell_line.count("Sperm") > 0
            or cell_line.count("Squamous cell carcinoma ") > 0
            or cell_line.count("Suprapubic skin") > 0
            or cell_line.count("Synoviocytes") > 0
            or cell_line.count("Thyroid carcinoma") > 0
            or cell_line.count("Thyroid gland") > 0
            or cell_line.count("Tibial arteries") > 0
            or cell_line.count("Tibial nerve") > 0
            or cell_line.count("Trophoblast stem cells") > 0
            or cell_line.count("Uterine epithelial cells") > 0
            or cell_line.count("Uveal melanoma") > 0
            or cell_line.count("Uvss1") > 0
            or cell_line.count("White adipose tissue") > 0
            or cell_line.count("Xenograft") > 0
            or cell_line.count("tomach") > 0
        ):
            keep = False

    cell_line = cell_line.upper()

    if keep:
        if (
            treatment.count("3-MB-PP1") > 0
            or treatment.count("5-FU") > 0
            or treatment.count("arsenic") > 0
            or treatment.count("A485") > 0
            or treatment.count("androgen") > 0
            or treatment.count("AID-treated") > 0
            or treatment.count("Arvanil") > 0
            or treatment.count("Auxin") > 0
            or treatment.count("Ad GATA") > 0
            or treatment.count("charcoal") > 0
            or treatment.count("BPA") > 0
            or treatment.count("Belinostat") > 0
            or treatment.count("Bicalutamide") > 0
            or treatment.count("BRD") > 0
            or treatment.count("Bica") > 0
            or treatment.count("calcitriol") > 0
            or treatment.count("clobetasol") > 0
            or treatment.count("Crizotinib") > 0
            or treatment.count("CBL0137") > 0
            or treatment.count("CsA") > 0
            or treatment.count("CPI360") > 0
            or treatment.count("CPA") > 0
            or treatment.count("CD532") > 0
            or treatment.count("Darolutamide") > 0
            or treatment.count("DHT") > 0
            or treatment.count("deprivation") > 0
            or treatment.count("DEX") > 0
            or treatment.count("Dex") > 0
            or treatment.count("dex") > 0
            or treatment.count("Dexamethasone") > 0
            or treatment.count("dexamethasone") > 0
            or treatment.count("Dinaciclib") > 0
            or (treatment.count("Dox") > 0 and treatment.count("No Dox") == 0)
            or treatment.count("DOX") > 0
            or treatment.count("estrogen") > 0
            or treatment.count("etoposide") > 0
            or treatment.count("electroporat") > 0
            or treatment.count("nzalutamide") > 0
            or treatment.count("Enz") > 0
            or treatment.count("E2") > 0
            or treatment.count("EGF") > 0
            or treatment.count("Estradiol") > 0
            or treatment.count("forskolin") > 0
            or treatment.count("flavopiridol") > 0
            or treatment.count("Genistein") > 0
            or treatment.count("glutamine") > 0
            or treatment.count("gSI") > 0
            or treatment.count("Gy") > 0
            or treatment.count("H2O2") > 0
            or (treatment.count("hour") > 0 and treatment.count("0hour") == 0)
            or treatment.count("hydrogen peroxide") > 0
            or treatment.count("hydroxytamoxifen") > 0
            or treatment.count("hormone-deprived") > 0
            or treatment.count("induced") > 0
            or treatment.count("infected") > 0
            or treatment.count("Il1b") > 0
            or treatment.count("ICBP112") > 0
            or treatment.count("I-BRD9") > 0
            or treatment.count("IFN") > 0
            or treatment.count("IFNg") > 0
            or treatment.count("IL-1") > 0
            or treatment.count("Imatinib") > 0
            or treatment.count("JQ1") > 0
            or (treatment.count("knockdown") > 0 and treatment.count("control") == 0)
            or treatment.count("LD100") > 0
            or treatment.count("lenti") > 0
            or treatment.count("LPS") > 0
            or treatment.count("LSD1") > 0
            or treatment.count("M tri-DAP") > 0
            or treatment.count("MNNG") > 0
            or treatment.count("Mifepristone") > 0
            or treatment.count("Nicotine") > 0
            or treatment.count("Nutlin") > 0
            or treatment.count("Over-expressing") > 0
            or treatment.count("OSMI-2") > 0
            or treatment.count("Pam2CSK4") > 0
            or treatment.count("Poly I:C") > 0
            or treatment.count("PMA") > 0
            or treatment.count("pravastatin") > 0
            or (treatment.count("sh") > 0 and treatment.count("shCtrl") == 0)
            or treatment.count("starvation") > 0
            or treatment.count("statin") > 0
            or (
                treatment.count("siRNA") > 0
                and treatment.count("control siRNA") == 0
                and treatment.count("Non-Target") == 0
            )
            or (
                treatment.count("Tetracycline") > 0
                and treatment.count("No Tetracycline") == 0
            )
            or treatment.count("TPA") > 0
            or (
                treatment.count("transfected with") > 0
                and treatment.count("empty vector") == 0
            )
            or treatment.count("transduced") > 0
            or treatment.count("trametinib") > 0
            or treatment.count("Tg") > 0
            or treatment.count("R1881") > 0
            or treatment.count("RU486") > 0
            or (treatment.count("siRNA") > 0 and treatment.count("NTsiRNA") == 0)
            or treatment.count("TAE684") > 0
            or treatment.count("TGF") > 0
            or treatment.count("TNF") > 0
            or treatment.count("Treated with") > 0
            or treatment.count("Shh") > 0
            or treatment.count("UPF-1069") > 0
            or treatment.count("Vemurafenib") > 0
            or treatment.count("VEGF") > 0
            or treatment.count("VTP") > 0
            or treatment.count("XY018") > 0
        ):
            keep = False

    if keep:
        if (
            source.count("AICAR") > 0
            or (
                source.count("AR_ChIP-seq_DU145 AR") > 0
                and source != "AR_ChIP-seq_DU145 AR"
            )
            or source.count("CMut") > 0
            or source.count("CS1AN") > 0
            or source.count("CSB_GFP") > 0
            or source.count("Chromatin") > 0
            or source.count("dEC") > 0
            or source.count("Doxycycline") > 0
            or (source.count("Dox") > 0 and source.count("NoDox") == 0)
            or source.count("Dox treated") > 0
            or source.count("DNA_damage") > 0
            or source.count("engineered to express") > 0
            or source.count("ERRgama") > 0
            or source.count("ER:Ras") > 0
            or (
                source.count("FOXA1_ChIP-seq_DU145 AR") > 0
                and source != "FOXA1_ChIP-seq_DU145 AR"
            )
            or (source.count("infected with") > 0 and source.count("shGFP") == 0)
            or source.count(" KO ") > 0
            or source.count("knockdown") > 0
            or source.count("overexpress") > 0
            or source.count("pcDNA3") > 0
            or source.count("PU.1") > 0
            or source.count("tamoxifen") > 0
            or (
                source.count("treated") > 0
                and source.count("vehicle") == 0
                and source.count("DMSO") == 0
            )
            or source.count("xenograft") > 0
        ):
            keep = False

    if keep:
        if (
            description.count("-KD") > 0
            or description.count("ARRB1") > 0
            or description.count("ARmo") > 0
            or description.count("ARhi") > 0
            or description.count("-si") > 0
            or description.count("-tfs") > 0
            or description.count("-cst") > 0
            or (description.count("si") > 0 and description.count("siCon") == 0)
            or description.count("Bicalutamide") > 0
            or (
                description.count(" sh") > 0
                and description.count("shCtrl") == 0
                and description.count("shGFP") == 0
            )
            or description.count("-/-") > 0
            or description.count(" OE ") > 0
            or (description.count(" KO ") > 0 and description.count("EmptyVector") > 0)
            or description.count(" KO;") > 0
            or description.count("calcitriol") > 0
            or description.count("csFCS") > 0
            or description.count(" C70 ") > 0
            or description.count("CRISPR") > 0
            or description.count(" CSS ") > 0
            or description.count(" CSSplus") > 0
            or description.count(".del") > 0
            or description.count(" DAU") > 0
            or description.count("depleated") > 0
            or description.count("DHT") > 0
            or description.count("differentiated") > 0
            or description.count(" Dox ") > 0
            or description.count("Doxorubicin") > 0
            or description.count("expression") > 0
            or description.count(" E2 ") > 0
            or description.count("Enzalutamide") > 0
            or description.count("estradiol") > 0
            or description.count("foxa1-") > 0
            or description.count("FOXA1 FA") > 0
            or description.count("FOXA1 DSG") > 0
            or description.count("GSK") > 0
            or description.count("HA only") > 0
            or description.count("hormone-deprived") > 0
            or description.count("HIF-1b") > 0
            or description.count("HNF4G") > 0
            or description.count("ypoxia") > 0
            or description.count("IBET151") > 0
            or description.count("IFNgamma") > 0
            or description.count("IgG") > 0
            or description.count("IPBRD7") > 0
            or description.count("JQ1") > 0
            or description.count(" KO") > 0
            or description.count(" knockdown ") > 0
            or description.count("LNCaP HNF4G") > 0
            or description.count("LNCaP NMYC") > 0
            or description.count(" mut ") > 0
            or description.count("Mib") > 0
            or description.count("MUT8") > 0
            or description.count("-null") > 0
            or description.count("OMOMYC") > 0
            or description.count("p53-/-") > 0
            or description.count("p53-KO") > 0
            or description.count("Progesterone") > 0
            or description.count("1881") > 0
            or description.count("R5020") > 0
            or description.count("shaml") > 0
            or description.count("sip53") > 0
            or description.count("SR2211") > 0
            or description.count("SH4") > 0
            or description.count("sh3") > 0
            or description.count("ARv7") > 0
            or description.count("ARN20") > 0
            or (
                description.count("treated") > 0
                and description.count("vehicle") == 0
                and description.count("DMSO") == 0
            )
            or description.count("targeting enhancer") > 0
            or description.count("TNF") > 0
            or description.count("TPA") > 0
            or description.count("VP16") > 0
            or description.count("week") > 0
            or description.count("WT-COMB") > 0
        ):
            keep = False

    if keep:
        i = fo.write(
            "\t".join(
                [
                    id,
                    tf,
                    cell_line,
                    description,
                    treatment,
                    treatment_duration,
                    source,
                    n_mapped,
                    pct_mapped,
                    pct_dup,
                    n_peaks,
                ]
            )
            + "\n"
        )
    else:
        i = fo_rejected.write(
            "\t".join(
                [
                    id,
                    tf,
                    cell_line,
                    description,
                    treatment,
                    treatment_duration,
                    source,
                    n_mapped,
                    pct_mapped,
                    pct_dup,
                    n_peaks,
                ]
            )
            + "\n"
        )

fo.close()
f.close()
