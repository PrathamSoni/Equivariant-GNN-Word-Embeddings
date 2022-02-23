curl --output README.txt https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/README.txt

mkdir raw

for i in {1..9}
do
  curl --output raw/000$i.xml.gz https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed22n000$i.xml.gz
done
#
#for i in {10..99}
#do
#  curl --output raw/00$i.xml.gz https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed22n00$i.xml.gz
#done
#
#for i in {100..999}
#do
#  curl --output raw/0$i.xml.gz https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed22n0$i.xml.gz
#done
#
#for i in {1000..1114}
#do
#  curl --output raw/$i.xml.gz https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed22n$i.xml.gz
#done
cd raw
gzip -dk *.gz
