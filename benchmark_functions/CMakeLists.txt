# ----------------------------------------------------------------------------
#  Root CMake file for the folder test containing all benchmark functions
#
# ----------------------------------------------------------------------------

set(benchmarks_files
    BK1
    CL1
    Deb41
    Deb512a
    Deb512b
    Deb512c
    Deb513
    Deb521a
    Deb521b
    Deb53
    DG01
    DPAM1
    DTLZ1
    DTLZ2
    DTLZ3
    DTLZ4
    DTLZ5
    DTLZ6
    ex005
    Far1
    FES1
    FES2
    FES3
    Fonseca
    I1
    I2
    I3
    I4
    I5
    IKK1
    IM1
    Jin1
    Jin2
    Jin3
    Jin4
    Kursawe
    L1ZDT4
    L2ZDT1
    L2ZDT2
    L2ZDT3
    L2ZDT4
    L2ZDT6
    L3ZDT1
    L3ZDT2
    L3ZDT3
    L3ZDT4
    L3ZDT6
    LE1
    Lovison1
    Lovison2
    Lovison3
    Lovison4
    Lovison5
    Lovison6
    LRS1
    MHHM1
    MHHM2
    MLF1
    MLF2
    MOP1
    MOP2
    MOP3
    MOP4
    MOP5
    MOP6
    MOP7
    OKA1
    OKA2
    QV1
    Sch1
    SK1
    SK2
    SP1
    SSFYY1
    SSFYY2
    TKLY1
    VFM1
    VU1
    VU2
    WFG1
    WFG2
    WFG3
    WFG4
    WFG5
    WFG6
    WFG7
    WFG8
    WFG9
    ZDT1
    ZDT2
    ZDT3
    ZDT4
    ZDT6
    ZLT1
    )

# build executables
foreach(filename ${benchmarks_files})
    ADD_EXECUTABLE("${filename}_test" "${CMAKE_CURRENT_SOURCE_DIR}/${filename}.cpp")
    TARGET_LINK_LIBRARIES("${filename}_test"
        PUBLIC nomadAlgos nomadUtils nomadEval)
    target_include_directories("${filename}_test" PUBLIC ${CMAKE_SOURCE_DIR}/src)
endforeach()
