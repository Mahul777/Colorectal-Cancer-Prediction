import kfp
from kfp import dsl  # dsl module is used to define pipelines


# Components of the pipeline
def data_processing_op():
    return dsl.ContainerOp(
        name='data-processing',
        image='apoorv678/my-mlops-app:latest',  # Ensure this image exists in the registry
        command=['python', 'src/data_processing4.py'],  # Use forward slashes for paths
        file_outputs={
            'output': '/app/data/output.csv'  # Ensure the output path is correct inside the container
        }
    )


def model_training_op():
    return dsl.ContainerOp(
        name='model-training',
        image='apoorv678/my-mlops-app:latest',
        command=['python', 'src/model_training5.py'],  # Use forward slashes for paths
        file_outputs={
            'output': '/app/model/model.pkl'  # Ensure the output path is correct inside the container
        }
    )

# Pipeline starts here
@dsl.pipeline(
    name='MLOps Pipeline 2',
    description='An example MLOps pipeline that includes data processing and model training.'
)
def mlops_pipeline():
    # Define pipeline steps
    data_processing = data_processing_op()  # Declare the data processing step
    model_training = model_training_op()  # Declare the model training step
    model_training.after(data_processing)  # Ensure model training happens after data processing

# Compile the pipeline into a YAML file
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(mlops_pipeline, 'mlops_pipeline_2.yaml')
