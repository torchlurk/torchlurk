import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='torchlurk',  
     version='0.1',
     scripts=[] ,
     author="Yann Mentha",
     author_email="yann.mentha@gmail.com",
     description="A CNN visualization library for pytorch",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/ymentha14/Torchlurk",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
