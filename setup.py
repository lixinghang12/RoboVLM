import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(
    name="robovlm",  # Replace with your username
    version="0.0.1",
    author="Xinghang Li, Minghuan Liu, Hanbo Zhang, et al.",
    author_email="ericliuof97@gmail.com",
    description="Finetuning VLMs on Robot Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="<https://github.com/lixinghang12/RobotVLM>",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)