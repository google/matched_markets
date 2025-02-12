package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "examples",
    srcs = [
        "examples/data_simulator.py",
        "examples/salesandcost.py",
    ],
    data = [
        "csv/geoassignment.csv",
        "csv/salesandcost.csv",
    ],
    deps = [":geoexperiments"],
)

py_test(
    name = "test_salesandcost",
    size = "small",
    srcs = [
        "examples/salesandcost.py",
        "tests/test_salesandcost.py",
    ],
    data = [
        "csv/geoassignment.csv",
        "csv/salesandcost.csv",
    ],
)

py_library(
    name = "geoexperiments",
    srcs = [
        "methodology/base.py",
        "methodology/common_classes.py",
        "methodology/robust.py",
        "methodology/robust_iroas.py",
        "methodology/semantics.py",
        "methodology/tbr.py",
        "methodology/tbr_iroas.py",
        "methodology/tbrdiagnostics.py",
        "methodology/utils.py",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_tbr",
    size = "small",
    srcs = [
        "tests/test_tbr.py",
    ],
    deps = [
        ":examples",
        ":geoexperiments",
    ],
)

py_test(
    name = "test_robust",
    size = "medium",
    srcs = [
        "tests/test_robust.py",
    ],
    deps = [
        ":examples",
        ":geoexperiments",
        "//testing/pybase:parameterized",
    ],
)

py_test(
    name = "test_tbr_iroas",
    size = "small",
    srcs = [
        "tests/test_tbr_iroas.py",
    ],
    deps = [
        ":examples",
        ":geoexperiments",
    ],
)

py_test(
    name = "test_robust_iroas",
    size = "medium",
    srcs = [
        "tests/test_robust_iroas.py",
    ],
    deps = [
        ":examples",
        ":geoexperiments",
        "//testing/pybase:parameterized",
    ],
)

py_test(
    name = "test_semantics",
    size = "small",
    srcs = [
        "methodology/semantics.py",
        "tests/test_semantics.py",
    ],
)

py_test(
    name = "test_data_simulator",
    size = "small",
    srcs = [
        "tests/test_data_simulator.py",
    ],
    deps = [":examples"],
)

py_library(
    name = "geoxdiag_data",
    srcs = [
        "examples/geoxdiag_data.py",
    ],
    data = [
        "csv/geoxdiag_data.csv",
    ],
)

py_test(
    name = "test_geoxdiag_data",
    size = "small",
    srcs = [
        "tests/test_geoxdiag_data.py",
    ],
    deps = [":geoxdiag_data"],
)

py_test(
    name = "test_tbrdiagnostics",
    size = "small",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrdiagnostics.py",
    ],
    data = [
        "csv/outlier_dataset.csv",
    ],
    deps = [
        ":geoexperiments",
        ":geoxdiag_data",
    ],
)

py_test(
    name = "test_utils",
    size = "small",
    srcs = [
        "tests/test_utils.py",
    ],
    deps = [":geoexperiments"],
)

py_library(
    name = "tbrmatchedmarkets",
    srcs = [
        "methodology/geoeligibility.py",
        "methodology/heapdict.py",
        "methodology/tbrmatchedmarkets.py",
        "methodology/tbrmatchedmarkets_v2.py",
        "methodology/tbrmmdata.py",
        "methodology/tbrmmdesign.py",
        "methodology/tbrmmdesignparameters.py",
        "methodology/tbrmmdiagnostics.py",
        "methodology/tbrmmscore.py",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_tbrmatchedmarkets",
    size = "medium",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrmatchedmarkets.py",
    ],
    data = [
        "csv/salesandcost.csv",
    ],
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_tbrmatchedmarkets2",
    size = "medium",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrmatchedmarkets2.py",
    ],
    data = [
        "csv/salesandcost.csv",
    ],
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_tbrmatchedmarkets3",
    size = "medium",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrmatchedmarkets3.py",
    ],
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_tbrmmscore",
    size = "small",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrmmscore.py",
    ],
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_geoeligibility",
    size = "small",
    timeout = "moderate",
    srcs = [
        "tests/test_geoeligibility.py",
    ],
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_tbrmmdesign",
    size = "small",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrmmdesign.py",
    ],
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_tbrmmdesignparameters",
    size = "small",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrmmdesignparameters.py",
    ],
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_heapdict",
    size = "small",
    srcs = [
        "tests/test_heapdict.py",
    ],
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_tbrmmdiagnostics",
    size = "small",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrmmdiagnostics.py",
    ],
    deps = [
        ":tbrmatchedmarkets",
        "//testing/pybase:parameterized",
    ],
)

py_test(
    name = "test_tbrmmdata",
    size = "small",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrmmdata.py",
    ],
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_common_classes",
    size = "small",
    srcs = [
        "tests/test_common_classes.py",
    ],
    deps = [":geoexperiments"],
)
