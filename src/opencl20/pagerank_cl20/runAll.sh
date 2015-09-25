for f in ./input/csrMatrix/*
do
	echo "$f"
	./bin/x86_64/Release/pagerank20 $f
done
