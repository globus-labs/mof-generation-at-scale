### Steps to Run the Workflow

1. **Activate the Environment**
   - Enter the `mofa` environment by running:
     ```bash
     conda activate mofa
     ```

2. **Start Redis Server**
   - Launch the Redis server in the background:
     ```bash
     redis-server --daemonize yes
     ```

3. **Set Environment Variables**
   - Set the following environment variables for AWS access and Kafka servers:
     ```bash
        export OCTOPUS_AWS_ACCESS_KEY_ID=<Your_AWS_Access_Key_ID>
        export OCTOPUS_AWS_SECRET_ACCESS_KEY=<Your_AWS_Secret_Access_Key>
        export OCTOPUS_BOOTSTRAP_SERVERS=<Octopus_Servers>
     ```

### Running the Workflow

You have two options to run the workflow:

#### **Option 1: Standard Run**
   - Execute the workflow using the standard command:
     ```bash
     ./example-parallel-run.sh 2>&1 | tee example-parallel-run.log
     ```

#### **Option 2: Separate Execution for Server and Thinker**
   - First, run the server component in one terminal:
     ```bash
     ./example-parallel-run.sh "--launch-option server" 2>&1 | tee example-parallel-run-server.log
     ```
   - In another terminal, set the environment variables again and run the thinker component:
     ```bash
     export OCTOPUS_AWS_ACCESS_KEY_ID=<Your_AWS_Access_Key_ID>
     export OCTOPUS_AWS_SECRET_ACCESS_KEY=<Your_AWS_Secret_Access_Key>
     export OCTOPUS_BOOTSTRAP_SERVERS=<Octopus_Servers>

     ./example-parallel-run.sh "--launch-option thinker" 2>&1 | tee example-parallel-run-thinker.log
     ```
