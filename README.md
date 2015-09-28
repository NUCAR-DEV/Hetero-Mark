Hetero-Mark

HSA benchmarks

Compilation guide

    To compile without debug
        cmake .

    To compile with debug
        cmake -DCMAKE_BUILD_TYPE=Debug .

Development guide

    The skeleton code for new benchmark is available
    in src/template directory.

    Make sure to add new benchmark dir to CmakeList.txt
    file in src/, otherwise new benchmark won't be 
    compiled with others.

