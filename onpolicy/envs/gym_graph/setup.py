import setuptools

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setuptools.setup(
        name='graph_env',
        version='2.0',
        description='Graph Environment for ASIST',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://gitlab.com/cmu_asist/gym_graph',
        packages=setuptools.find_packages(include=['graph_env']),
        python_requires='>=3.8',
)
