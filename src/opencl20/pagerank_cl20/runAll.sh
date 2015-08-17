for f in ./input/csrMatrix/*
do
	echo "$f"
	./bin/x86_64/Release/pagerank $f
done
