import setuptools

setuptools.setup(
    ext_modules=[
        setuptools.Extension(
            "wrapt._wrappers",
            sources=["src/wrapt/_wrappers.c"],
            optional=False,
        )
    ]
)
