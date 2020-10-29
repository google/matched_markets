package(default_visibility = ["//visibility:public"])

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
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
)

py_library(
    name = "geoexperiments",
    srcs = [
        "methodology/common_classes.py",
        "methodology/semantics.py",
        "methodology/tbr.py",
        "methodology/tbr_iroas.py",
        "methodology/tbrdiagnostics.py",
        "methodology/utils.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_tbr",
    size = "small",
    srcs = [
        "tests/test_tbr.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":examples",
        ":geoexperiments",
    ],
)

py_test(
    name = "test_tbr_iroas",
    size = "small",
    srcs = [
        "tests/test_tbr_iroas.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":examples",
        ":geoexperiments",
    ],
)

py_test(
    name = "test_semantics",
    size = "small",
    srcs = [
        "methodology/semantics.py",
        "tests/test_semantics.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
)

py_test(
    name = "test_data_simulator",
    size = "small",
    srcs = [
        "tests/test_data_simulator.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
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
    srcs_version = "PY3",
)

py_test(
    name = "test_geoxdiag_data",
    size = "small",
    srcs = [
        "tests/test_geoxdiag_data.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":geoxdiag_data"],
)

py_test(
    name = "test_tbrdiagnostics",
    size = "small",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrdiagnostics.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":geoexperiments"],
)

py_library(
    name = "tbrmatchedmarkets",
    srcs = [
        "methodology/geoeligibility.py",
        "methodology/heapdict.py",
        "methodology/tbrmatchedmarkets.py",
        "methodology/tbrmmdata.py",
        "methodology/tbrmmdesign.py",
        "methodology/tbrmmdesignparameters.py",
        "methodology/tbrmmdiagnostics.py",
        "methodology/tbrmmscore.py",
    ],
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_tbrmatchedmarkets3",
    size = "medium",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrmatchedmarkets3.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_tbrmmscore",
    size = "small",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrmmscore.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_geoeligibility",
    size = "small",
    timeout = "moderate",
    srcs = [
        "tests/test_geoeligibility.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_tbrmmdesign",
    size = "small",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrmmdesign.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_tbrmmdesignparameters",
    size = "small",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrmmdesignparameters.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_heapdict",
    size = "small",
    srcs = [
        "tests/test_heapdict.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_tbrmmdiagnostics",
    size = "small",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrmmdiagnostics.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_tbrmmdata",
    size = "small",
    timeout = "moderate",
    srcs = [
        "tests/test_tbrmmdata.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tbrmatchedmarkets"],
)

py_test(
    name = "test_common_classes",
    size = "small",
    srcs = [
        "tests/test_common_classes.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":geoexperiments"],
)
