for i in {0..15}; do
    python data/vaccine/chunking/chunking.py $i &
done

