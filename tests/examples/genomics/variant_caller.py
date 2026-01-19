from collections import defaultdict

import pysam


class VariantCaller:
    def __init__(
        self,
        variant_caller="gatk",
        reference_genome="hg38",
        min_quality_score=30,
        min_read_depth=10,
    ):
        self.variant_caller = variant_caller
        self.reference_genome = reference_genome
        self.min_quality_score = min_quality_score
        self.min_read_depth = min_read_depth

    def call_variants(self, bam_file, output_vcf):
        """Call genetic variants from alignment file"""
        # Read alignment file
        bamfile = pysam.AlignmentFile(bam_file, "rb")

        variants = []
        for pileupcolumn in bamfile.pileup():
            # Filter by depth
            if pileupcolumn.n < self.min_read_depth:
                continue

            # Count alleles at this position
            allele_counts = defaultdict(int)
            for pileupread in pileupcolumn.pileups:
                if not pileupread.is_del and not pileupread.is_refskip:
                    base = pileupread.alignment.query_sequence[pileupread.query_position]
                    allele_counts[base] += 1

            # Call variant if alternate allele frequency > 20%
            total_reads = sum(allele_counts.values())
            for allele, count in allele_counts.items():
                freq = count / total_reads
                if freq > 0.2 and freq < 0.8:
                    quality = min(99, count * 2)
                    if quality >= self.min_quality_score:
                        variants.append(
                            {
                                "chr": pileupcolumn.reference_name,
                                "pos": pileupcolumn.pos,
                                "ref": "N",
                                "alt": allele,
                                "quality": quality,
                                "depth": total_reads,
                            }
                        )

        return variants


# Example usage
if __name__ == "__main__":
    caller = VariantCaller(min_quality_score=30, min_read_depth=10)
    variants = caller.call_variants("sample.bam", "output.vcf")
    print(f"Found {len(variants)} high-quality variants")
