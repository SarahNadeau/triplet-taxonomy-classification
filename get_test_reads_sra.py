from Bio import SeqIO
import os
import numpy as np
from numpy.random import randint


# this function gets a random sample of test reads from specified fastq files
def get_test_reads(file, kmer_len, num_reads):
    seq_count = 0
    seq_num = 0
    unclassified_reads = []
    for record in SeqIO.parse(file, 'fastq'):
        seq_count += 1

    for record in SeqIO.parse(file, 'fastq'):
        if seq_num < seq_count and seq_num < num_reads:
            len_seq = len(record.seq)
            if len_seq > kmer_len:
                rand_seq_start = randint(0, len_seq - kmer_len)
                unclassified_reads.append(str(record.seq[rand_seq_start: rand_seq_start + kmer_len]))
                seq_num += 1
    return unclassified_reads


# test read taxonomy is hardcoded here (taxonomy is available on NCBI site)
def get_test_tax(file):
    SRR_tax_dict = {'SRR6357671': ['Bacteria', 'Actinobacteria', 'Micrococcales', 'Micrococcaceae', 'Zhihengliuella','Zhihengliuella sp. ISTPL4'],
                    'SRR5324459': ['Bacteria', 'Actinobacteria', 'Acidimicrobiia', 'Acidimicrobiales', 'Microthrixaceae', 'Candidatus Microthrix sp. UBA1665'],
                    'SRR006904': ['Bacteria', 'Actinobacteria', 'Streptosporangiales', 'Streptosporangiaceae', 'Astrosporangium', 'Astrosporangium hypotensionis'],
                    'SRR005145': ['Bacteria', 'Actinobacteria', 'Coriobacteriales', 'Atopobiaceae', 'Atopobium', 'Atopobium rimae'],
                    'SRR6045835': ['Bacteria', 'Actinobacteria', 'Corynebacteriales', 'Mycobacteriaceae', 'Mycobacterium', 'Mycobacterium tuberculosis complex']
                    }
    # SRR6357671, SRR006904, and SRR5324459 do not exist in Actinobacteria complete reference genomes
    # Sequencing technologies used: HiSeq, HiSeq, 454 GS FLX, 454 GS FLX, MiSeq
    for record in SeqIO.parse(file, 'fastq'):
        SRR = record.id.split('.')[0]
        true_tax = SRR_tax_dict[SRR]
    return true_tax


def main():
    # get one exact test sequence from each
    kmer_len = 100
    num_reads = 5

    file_list = []
    for file_name in os.listdir("SRA_Test_Sequences"):
        if file_name.endswith(".fastq"):
            file_name = os.path.join("SRA_Test_Sequences", file_name)
            file_list.append(file_name)

    for file in file_list:
        unclassified_reads = get_test_reads(file, kmer_len, num_reads)
        print(unclassified_reads)
        true_tax = get_test_tax(file)
        print(true_tax)


if __name__ == "__main__":
    main()






