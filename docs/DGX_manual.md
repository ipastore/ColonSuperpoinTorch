# Basic DGX info, rules, and guidelines

Table of Contents

[1\. MAIN “RULES”](#1.-main-“rules”)

[2\. DGX-1 system](#2.-dgx-1-system)

[2.1 Create a DGX-1 account](#2.1-create-a-dgx-1-account)

[2.2 Connect to DGX-1 using SSH](#2.2-connect-to-dgx-1-using-ssh)

[2.3 Transfer data to/from DGX-1](#2.3-transfer-data-to/from-dgx-1)

[3\. Docker image](#3.-docker-image)

[3.1 Creating your custom image](#3.1-creating-your-custom-image)

[4\. How to launch jobs using SLURM](#4.-how-to-launch-jobs-using-slurm)

[4.1 SRUN](#4.1-srun)

[4.2 SBATCH](#4.2-sbatch)

[4.3 Other useful SLURM commands](#4.3-other-useful-slurm-commands)

[5\. SLURM \+CONDA/MAMBA](#5.-slurm-+conda/mamba)

[6\. Known errors](#6.-known-errors)

[Not enough space](#not-enough-space)

[Job stuck in the queue despite having available resources](#job-stuck-in-the-queue-despite-having-available-resources)

[Srun error:  Permission denied to /raid/enroot-cache/group-18000/\*:](#srun-error:-permission-denied-to-/raid/enroot-cache/group-18000/*:)

*This document assumes the use of a Command Line Interface (CLI) that runs Bash on a GNU/Linux environment (e.g. any standard terminal in Ubuntu). Compatibility with PowerShell and Windows Subsystem For Linux in Windows is not assured.*

Through this document, variables that will change for each reader will be written in green and enclosed with angle brackets, like *\<user\>* , and should be updated with each reader’s corresponding data.

# **1\. MAIN “RULES”** {#1.-main-“rules”}

These are some considerations that all DGX users should follow to ensure a correct use of the DGX.

1. **Everything that needs a lot of storage (e.g. all your results, datasets, data, and code) should be in your *storage* folder** (e.g. **/home/ropert/anacris/storage**).  
   This storage folder is a “symlink” to a folder in a very large drive (**/raid/**) :-) If it’s not created, please ask support to do it for you with a ticket to [I3A support](http://soporte.i3a.es/).  
   When connected to DGX-1, running in the command line this : 

```
<user>@dgx01:~$ ls -l /home/ropert/<user>/
```

   with your username instead of ***\<user\>***, should output something like this:

```
lrwxrwxrwx 1 root root 20 Nov 25 09:13 storage -> /raid/ropert/<user>
```

 


2. **SLURM is in charge of managing the DGX resources (e.g. GPUs, CPUs, and memory) avoiding conflicts. All jobs should be launched through SLURM. *Do not directly use Docker to avoid conflicts with SLURM.***  
3. SLURM accumulates temporal files which can represent a big proportion of ***/raid/*** memory. From time to time, and without a running job, run from the command line interface

```
<user>@dgx01:~$ enroot remove --force $(enroot list)
<user>@dgx01:~$ rm -rf /raid/tmp/* 
```

If we see that certain users are not doing this, we will have to change the permissions and limits for these users and/or automatically delete containers from time to time.

Another folder that may require attention is  /root/.cache/torch/hub/ which may be storing an indefinite number of large files since it is the default directory for downloading checkpoints from [Torch Hub](https://pytorch.org/hub/). However, access to this folder requires sudo privileges.

Interesting “pointers” you may want/need to read:

* DOCKER official documentation: https://docs.docker.com/  
* NVIDIA container catalog: https://ngc.nvidia.com/catalog/containers  
* SLURM official documentation: [https://slurm.schedmd.com/overview.html](https://slurm.schedmd.com/overview.html)

To perform a general “check” of disk usage from everyone, a **sudo user** can run the following command to see all users /raid storage

```
<user>@dgx01:~$ du -h --max-depth=1 /raid/ropert | sort -hrdu -hs /raid/ropert/* | awk '{print $1, $2}' | sort -nr
```

# **2\. DGX-1 system** {#2.-dgx-1-system}

The system is a NVIDIA DGX-1 with: 

* 8 x NVIDIA V100 GPUs. Each with 32 GBytes of vRAM;  
* 80 x CPUs cores;  
* 512 Gbytes of RAM;  
* 1 x virtual disk of 4 TBytes for storage (/raid/). It is made of 4 x HDD disks of  1 TByte each. 

The system runs DGX OS 5.5 (GNU/Linux), SLURM 20.02.4, and NVIDIA drivers version 470.182.03 compatible to run  CUDA 11.4 libraries The CUDA SDK compiler is 10.1, outdated with respect to the drivers and other development tools (GCC, PyTorch…). Nevertheless, you can install a more recent version in your Docker container, as will be explained in the [Docker section](#3.-docker-image).  
The system’s last update was in July 2023\.

## 2.1 Create a DGX-1 account {#2.1-create-a-dgx-1-account}

To connect to the DGX-1 you will first need to request an account to I3A support following this steps:

1) Log-in to [http://soporte.i3a.es/](http://soporte.i3a.es/) with your official email user (without @unizar.es) and your email password.  
2) Create a ticket clicking on “Crear una petición”, and fill with the following information:  
* *Categoría*: HERMES \>\> Petición de Cuenta  
* *Título*: Solicitud acceso a la NVIDIA DGX-1  
* *Descripción*: 

  Nombre: \<your-name\>

  Correo: \<your-email\>

  Grupo: Grupo de robótica, percepción y tiempo real.

  Departamento: Departamento de Informática e Ingeniería de Sistemas.

  Área: Ingeniería de Sistemas y Automática.

  Puesto: \<your-position\>

  Motivo de creación de la cuenta: Uso de la DGX1.

  Duración estimada de la misma: \<Whatever time in years\> años.

  Comentarios *(just copy the following text)*: 

  **“Añadir al grupo docker para usar docker en la DGX1 y crear el acceso directo al “storage” en el home de la cuenta”**

3) *(Optional)* You probably also want to **create an account on the [NVIDIA NGC](https://ngc.nvidia.com/signin)** to access the docker catalog, examples, etc… (although it is not required to download docker images).  
   **If you work in the Endomapper project, once you created your NVIDIA account you can pass your username and email associated with it to Ana Cris or Luis Riazuelo to include your user in the Endomapper account.**

When you receive notification that your account has been created, ensure to fulfill [rule 1 from Main “RULES”](#1.-main-“rules”).

## 2.2 Connect to DGX-1 using SSH {#2.2-connect-to-dgx-1-using-ssh}

Once your account has been created, you can connect to the DGX-1 using [ssh](https://www.man7.org/linux/man-pages/man1/ssh.1.html) from the CLI.   
The DGX-1 has connections restricted to the port 3003, therefore you will have to use the command

```
<user>@<your-pc>:~$ ssh -p 3003 <your-dgx-user>@155.210.134.17  
```

## 2.3 Transfer data to/from DGX-1  {#2.3-transfer-data-to/from-dgx-1}

To copy data to the DGX, you can use several different tools (e.g. *Github*, *SCP, sftp* and *rsync).*  
You can also use an app with windows interface if you prefer, there are many. For example for windows and mac you could use [cyberduck](https://cyberduck.io/download)*.*  
One of the methods to move data is the [scp CLI command](https://www.man7.org/linux/man-pages/man1/scp.1.html). The fastest way to do this is from a terminal in ***your machine (not in the DGX-1)*** : 

```
<user>@<your-pc>:~$ scp -P 3003 <ORIGIN-PATH> <DESTINATION-PATH>
```

(Beware that with *scp* the input port flag is capital *\-P* rather than minuscule *\-p* ).  
For example, to move data

* from YOUR COMPUTER → to the DGX1,

```
<user>@<your-pc>:~$ scp -P 3003 ~/Downloads/HOW-TO_USE-DGX.pdf <dgx-user>@155.210.134.17:~/storage/.
```

* and from DGX1 → YOUR COMPUTER,

```
<user>@<your-pc>:~$ scp -P 3003 <dgx-user>@155.210.134.17:~/storage/HOW-TO_USE-DGX.pdf  ~/Downloads/. 
```

# **3\. Docker image** {#3.-docker-image}

We recommend using NVIDIA images. You can find them in the [NVIDIA catalog](https://catalog.ngc.nvidia.com/containers) . When choosing a version of the image ensure that it is compatible with [current DGX NVIDIA drivers](https://docs.google.com/document/d/1pCshNhWjaMkQDQMPj4sMSG30-APN0gsC5x6c6aKvNAU/edit?pli=1#heading=h.9t9d48ij34yj).

* The NVIDIA driver installed in the DGX (version 470.182.03) is compatible with up to CUDA 11.4. In addition, the DGX has an integrated forward compatibility mode that makes this driver compatible with the latest 11.x and 12.x. versions. You can read more about this feature [here](https://docs.nvidia.com/deploy/cuda-compatibility/#forward-compatibility-title).  
* For the [NVIDIA Pytorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags), you can check the NVIDIA driver requirements for each container version in  [this compatibility matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html). As of May 2024, all the latest versions (24.xx containers) are compatible with the DGX using the forward compatibility mode. The latest version that can run without this mode using the installed driver is 21.10  
* If you use a different container that requires GPU support, make sure that the container runs a CUDA version either forward or backward compatible with the installed driver. See the [driver release notes](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-470-182-03/index.html) and the [NVIDIA CUDA compatibility page](https://docs.nvidia.com/deploy/cuda-compatibility/) if you need more information.

If you are working with default containers, using SLURM you don’t need to download images beforehand, just check their catalog path.

## 3.1 Creating your custom image  {#3.1-creating-your-custom-image}

If you are going to launch the same script several times or you require several dependencies to run your code, you should modify the container to save time. Starting with an image from NVIDIA’s catalog, follow [this tutorial](https://phoenixnap.com/kb/how-to-commit-changes-to-docker-image) **in your local machine** to modify a Docker image and save the changes.

After creating your custom image, there are two ways to use it with SLURM:

* Convert it locally **using NVIDIA enroot**:   
1)  Install enroot on your computer following the repository instructions: [https://github.com/NVIDIA/enroot/blob/master/doc/installation.md](https://github.com/NVIDIA/enroot/blob/master/doc/installation.md)  
2) Use [enroot import](https://github.com/NVIDIA/enroot/blob/master/doc/cmd/import.md) to convert your docker image to an enroot image. For example

```
<user>@<your-pc>:~$ enroot import dockerd://<imagehash>
```

3) Transfer the created .sqsh file to the DGX server and use it with srun’s flag (e.g. \--container-image=/raid/ropert/\<user\>/\<path to image\>/myimage.sqsh), as explained in [srun’s section](#4.1-srun).

* Upload it to NVIDIA NGC and let SLURM convert it (**only if you have a NGC account linked to NGC endomapper project**):  
1) *(Optional)* Login in the NGC catalog and, in the private registry, Containers \>\> Create. Fill in the information and save the container. *If you skip this step, when pushing the image will be created automatically*.  
2) Go to the setup in the NGC catalog (Top right of the web, https://ngc.nvidia.com/setup/api-key)\>\> Get API Key \>\> Generate API Key \>\> Copy the key and save it for later  
3) If you didn’t while building it, tag your custom image with Endomapper repository info. (Change the command with your information):

```
<user>@<your-pc>:~$ docker tag <image_id> nvcr.io/endomapper/<name_of_the_container>:<tag_number>
```

	

4) Log to NVIDIA catalog from the command line with your username and password:

```
<user>@<your-pc>:~$ docker login nvcr.io
```

5) Push the image:

```
<user>@<your-pc>:~$ docker push nvcr.io/endomapper/<name_of_the_container>:<tag_number>
```

6) Now go to the DGX, in home, if it doesn’t exist create (folders and file): “.config/enroot/.credentials” . Inside “.credentials” fill with this:

```
# NVIDIA GPU Cloud (both endpoints are required)
machine nvcr.io login $oauthtoken password <YOUR_API_NGC_KEY>
machine authn.nvidia.com login $oauthtoken password <YOUR_API_NGC_KEY>
```

7) After that launch *srun,* as explained in [srun’s section](#4.1-srun), with the container flag directing to your image (e.g.  
   *\--container-image=nvcr.io\#endomapper/\<name\_of\_the\_container\>:\<tag\_number\>*).  
   

# 4\. How to launch jobs using SLURM {#4.-how-to-launch-jobs-using-slurm}

***This should be the only way to launch jobs, otherwise processes may fail due to resource conflicts.***

DGX-1 process can be launched using SLURM’s commands [*srun*](https://slurm.schedmd.com/srun.html) and [*sbatch*](https://slurm.schedmd.com/sbatch.html).  *Srun* will launch a *step* which adds your job to SLURM’s queue to execute it when enough resources are available. *Sbatch* is used to define an environment to launch one or more *srun* steps. Each *step* is made of one or more *tasks* running concurrently the same program.

Before launching a job you need to:

1. Import your code and data to the DGX-1 machine (see [2.3 Transfer data to/from DGX-1 machine](#2.3-transfer-data-to/from-dgx-1) ). **Everything must be stored under YOUR storage folder, as explained in [Main “RULES”](#1.-main-“rules”)** .  
2. Create a bash script, e.g. **train.sh,** which will launch your desired processes.

In **the executing script,** 

* first install the requirements (e.g., *pip3 install* …)¹ if there is any requirement needed that is not already installed in the docker image or in your Conda environment;   
* and then, execute your things (e.g., *python3 …*)


For example train.sh could be:

```
cd  /workspace/ialonso/Semi-Seg/code 
pip3 install -r requirements.txt 
python3 train_my_model.py
```

or if you use a custom image with Conda and a training script with argument parser:

```
#!/bin/bash
devices=$1
config_path=$2
source /opt/conda/etc/profile.d/conda.sh

conda activate myenvironment
cd /workspace/tberriel/my_git_repo

python3 train.py runs/$config_path/config.yaml --wandb --gradient_clip --devices $devices 
```

**If you are going to launch the same script several times, you should use a custom docker image. [See section 3](#3.-docker-image).**   

¹ *If some apt-get is interactive, add to your script: DEBIAN\_FRONTEND="noninteractive" apt-get install \--yes*

## 4.1 SRUN {#4.1-srun}

An example of the [*srun* command](https://slurm.schedmd.com/srun.html), launched from ***ialonso*** user, would be:

```
ialonso@dgx01:~$ srun --job-name=training --output=srun_log.out  -N 1 --gres=gpu:1 --mem-per-cpu=15G --container-mounts=/raid/ropert/ialonso/Semi-Seg/:/workspace/ialonso/Semi-Seg   --container-workdir=/workspace/ialonso --container-image=nvcr.io#nvidia/pytorch:20.10-py3 sh /workspace/ialonso/Semi-Seg/train.sh &
```

The command flags and parameters can be grouped in the following categories : `generic flags,` `resource flags,` `mount flags,` and  `script launch.`

`Generic flags` define your job’s name and  where to store your logs.  
*\--job-name=\<name\>* :  *(Optional)* Name of the job when listed in *squeue*. By default it will use the first word of  `script launch`.  
*\--output=\<output\_file\>* : *(Optional)* Log file containing the standard output of the launched script. Output text from your script will be saved inside \<output\_file\>. *By default both standard output and standard error are directed to the same file.*  
*\--error=\<error\_file\>* : *(Optional)* Log file containing the standard error of the launched script. Error text from your script will be saved here. If **\--error** is not specified, both stdout and stderr will be directed to the file specified by **\--output**.

`Resource flags` define the resources required to run your job. If the selected resources are available, the job status in the queue will change from ***Pending*** to ***Running***.   
*\-N* \<n\_nodes\> : Nodes to use. DGX-1 has only 1 node, hence it should always be \-N 1.    
*\--ntasks=\<n\_tasks\>* :  *(Optional)* Specifies how many tasks to run, and requests that ***srun*** allocates resources for \<n\_tasks\> tasks. Default is one task per node. This flag is only needed *for multi-GPU training with Pytorch Lightning,* when *\--ntasks should match the number of GPUs requested.*  
*\--gres=gpu:\<n\_gpus\>* :  *(Optional)* where *\<n\_gpus\>* is the number of GPUs required for the job. If the flag is not added, no GPU will be allocated.  
*\--cpus-per-task=\<n\_cpus\>* :  *(Optional)* Number of CPUs allocated per process. Default is two CPUs. The CPUs allocated will be \--ntasks \* \--cpus-per-task. *Although for newer versions of SLURM (\>22.05) when running srun from within sbatch, \--cpus-per-task is not inherited by srun; the DGX-1 is running with version 20.02.*  
*\--mem-per-cpu=\<size\>\[units\]* : *(Optional)* RAM memory required per allocated CPU.  Memory allocated will be  \--ntasks \* \--cpus-per-task \* \--mem-per-cpu. Default units in megabytes, different units can be specified using suffix \[K|M|G|T\]. *Current DGX-1 configuration allocates individual processors to jobs, therefore SLURM’s documentation recommends using **\--mem-per-cpu** **instead of \--mem**.*   
*\--mem=\<size\>\[units\]* : *(Deprecated)* RAM memory required per node, default units in megabytes, different units can be specified using suffix \[K|M|G|T\]. *Current DGX-1 configuration does not allocate a whole node per job, therefore SLURM’s documentation recommends using **\--mem-per-cpu instead of \--mem**.*  

`Mount flags` define which container to launch, and how to connect it to your DGX-1 storage.  
*\--container-mounts=\<src\_dir\>:\<tar\_dir\>* : Bind mount inside the container.  The directory \<src\_dir\> will be accessible as \<tar\_dir\> inside the container.  
*\--container-workdir=\<path\>*: path (inside the container) to set as the working directory  
*\--container-image=\<image\>*: Docker image to use for the container filesystem. Can be either a container from [NVIDIA catalog](https://catalog.ngc.nvidia.com/containers)² or a local image squashed with NVIDIA Enroot (See [Sec 3](#3.-docker-image) ).

`Script launch` is just:

1) the script interpreter, which defines the compiler to read your script. The example uses Bourne Shell “sh”, but you can also use Bash “/bin/bash”, or others.  
2) and the path to the script. In the example above the path (`/workspace/ialonso/Semi-Seg/train.sh`) uses the absolute path to the target folder in *\--container-mounts*. If your script takes any argument, they would go after the script name.

Finally, an **ampersand ‘&’** is added at the end of the command **to execute it asynchronously** launching the script in the background **without blocking your CLI. If** you **launch** *srun* **inside *sbatch*** there is **no need for** the **ampersand ‘&’ .**

**An additional example of srun** is:

```
tberriel@dgx01:~$ srun -N 1 --cpus-per-task=6 --gres=gpu:4 --mem-per-cpu=3G --ntasks-per-node=4  --output=srun_train.out --error=srun_train_full.err --container-mounts=/raid/ropert/tberriel/:/workspace/tberriel/   --container-workdir=/workspace/tberriel --container-image=/raid/ropert/tberriel/Docker/myimage.sqsh /bin/bash /workspace/tberriel/my_git_repo/train.sh 4 my_checkpoints/custom_model/  &
```

This example corresponds to the second example of train.sh, and launches a distributed training in 4 GPUs using a custom local docker image.

² *Use the address from the pull command and change the first dash (“/”) for a hash (“\#”), e.g. for [NVIDIA’s PyTorch image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) \<image\> would be “**nvcr.io\#nvidia/pytorch:20.10-py3**“.*

## 4.2 SBATCH {#4.2-sbatch}

[Sbatch](https://slurm.schedmd.com/sbatch.html) defines an environment for SLURM’s jobs and can launch one or more *srun* steps. Environment variables are the same as for the *srun* command. An example of an *sbatch* file called sbatch\_train.sh is:

```
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=srun_train.out
#SBATCH --error=srun_train.err
#SBATCH -N 1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=3G

srun --container-mounts=/raid/ropert/tberriel/Workspace/:/workspace/tberriel   --container-workdir=/workspace/tberriel --container-image=/raid/ropert/tberriel/Docker/myimage.sqsh /bin/bash /workspace/tberriel/my_git_repo/train.sh 4 msn_easy/mrrpt_8_32_2/ 1
```

And would be executed launching

```
tberriel@dgx01:~$ sbatch sbatch_train.sh
```

Inside the file you:

1) define your shell interpreter;  
2) then define the desired environment flags prepending \#SBATCH to (previously explained) *srun* flags. Sbatch will stop interpreting options after the first non-comment (no hash “\#”) non-whitespace line.  
3) Finally launch one or more srun commands with desired mount commands and launch script. If desired, *sbatch* flags can be overridden by *srun* specific flags..

**When using *sbatch*, there is no need for the** **ampersand ‘&’** at the end of *srun*. *Sbatch* already launches it in the background, and adding the ampersand would have a different impact.

## 4.3 Other useful SLURM commands {#4.3-other-useful-slurm-commands}

* *squeue*:  shows information about the jobs that have been scheduled by SLURM (whether they are currently running, in the waitlist etc.). For a more detailed control over the information displayed by *squeue*, we can use its format options (flag \-o) to print further useful details. For example, with the command


```
<user>@dgx01:~$squeue -o"%.7i %.10P %.10j %.8u %.2t %.8M %.6C %.10m %.13b" 
```


  we will also display the following information:

  * \-C: the requested number of CPUs,  
  * \-m: requested memory,   
  * \-b: requested number of GPUs.

  The numeric values correspond to the number of characters used per column to display the information.

* scancel \<job\_id\> : to cancel or terminate one of *your* jobs scheduled with SLURM. You can retrieve your specific \<job\_id\> by executing squeue and searching for your job in the list. 

* scontrol show node : displays detailed and useful information about the DGX node, such as  
  * Gres: The total number of GPUs in the DGX node (there are *8 GPUs*).  
  * CfgTRES=cpu: The total number of CPUs in the DGX (there are *80 CPUs*).  
  * RealMemory: The total memory, in MB, on the DGX node (there are *490088 MB*).  
  * AllocMem: The total memory, in MB, currently allocated by jobs on the DGX.  
  * AllocTRES=cpu, mem, gres/gpu: The total number of CPUs, memory and number of GPUs currently in use on the DGX.  
* scontrol show job \<job\_id\> :  displays detailed information about the job with ID \<job\_id\>, including used resources and mounting directories.

# 5\. SLURM \+CONDA/MAMBA {#5.-slurm-+conda/mamba}

Finally, for a short tutorial on how to use SLURM together with conda/mamba environments look at Javi Tirado’s [wonderful guide](https://docs.google.com/document/d/1OEMCBf7V0G2wncTj3SC_oKlsVDq-xoP-Qelz7_iHyxs/edit?usp=sharing).

# 6\. Known errors {#6.-known-errors}

### Not enough space {#not-enough-space}

---

* Delete the squashed images in the folder /run/pyxis/\<user\_id\>. When using images from NGC catalog pyxis stores them in this folder, which has a limit of 51 G byte. To find your \<user\_id\>, run `“ll”` or `“ls -l”` command and look at the folders’ owners.  
* Remove temporal files generated by SLURM. Without a running job, run from the command line interface:

```
<user>@dgx01:~$ enroot remove --force $(enroot list)
<user>@dgx01:~$ rm -rf /raid/tmp/* 
```

### Job stuck in the queue despite having available resources {#job-stuck-in-the-queue-despite-having-available-resources}

👏Thanks to Sergio Izquierdo for solving this problem  
---

**Possible solution: add flag *\--gres-flags=disable-binding* to *srun* command.**    
This is because SLURM’s configuration assigns to each GPU a range of CPUs

```
Name=gpu File=/dev/nvidia0 CPUs=0-9,40-49
Name=gpu File=/dev/nvidia1 CPUs=0-9,40-49
Name=gpu File=/dev/nvidia2 CPUs=10-19,50-59
Name=gpu File=/dev/nvidia3 CPUs=10-19,50-59
Name=gpu File=/dev/nvidia4 CPUs=20-29,60-69
Name=gpu File=/dev/nvidia5 CPUs=20-29,60-69
Name=gpu File=/dev/nvidia6 CPUs=30-39,70-79
Name=gpu File=/dev/nvidia7 CPUs=30-39,70-79
```

Nevertheless, there is an overlap between GPUs. Hence, when there is a high usage of CPUs it can happen that the number of requested CPUs is available, but they are not available inside the interval assigned to the available GPUs. This range can be ignored using the flag *\--gres-flags=disable-binding*  in the *srun* command, although it may result in a lower performance.

### Srun error:  Permission denied to `/raid/enroot-cache/group-18000/*`:  {#srun-error:-permission-denied-to-/raid/enroot-cache/group-18000/*:}

👏Thanks to Javier Tirado for solving this problem.  
This error is also thoroughly explained in Section 5.2 of the [SLURM+MAMBA Tutorial](https://docs.google.com/document/d/1OEMCBf7V0G2wncTj3SC_oKlsVDq-xoP-Qelz7_iHyxs/edit?usp=sharing).   
---

It may be the case that Pyxis (SLURM’s containers manager extension) is generating cache layers that can be used only by the first user that loaded a given docker image. When other users try to use the same image (or a different image which shares layers with the first image), Pyxis will try to reuse the cache but will get the access denied due to lack of permissions.  
In this situation the error will look something like this:

```
pyxis: importing docker image ...
slurmstepd: error: pyxis: child 185257 failed with error code: 1
slurmstepd: error: pyxis: failed to import docker image
slurmstepd: error: pyxis: printing contents of log file ...
slurmstepd: error: pyxis:     [INFO] Querying registry for permission grant
slurmstepd: error: pyxis:     [INFO] Authenticating with user: <anonymous>
slurmstepd: error: pyxis:     [INFO] Authentication succeeded
slurmstepd: error: pyxis:     [INFO] Fetching image manifest list
slurmstepd: error: pyxis:     [INFO] Fetching image manifest
slurmstepd: error: pyxis:     [INFO] Found all layers in cache
slurmstepd: error: pyxis:     [INFO] Extracting image layers...
slurmstepd: error: pyxis:     [INFO] Converting whiteouts...
slurmstepd: error: pyxis:     zstd: /raid/enroot-cache/group-18000/d9270b9c551dee8e10b082c3c5faf35c32b60ae377ab5d4d1d7e6a6ac23d9a45: Permission denied
slurmstepd: error: pyxis: couldn't start container
slurmstepd: error: pyxis: if the image has an unusual entrypoint, try using --no-container-entrypoint
slurmstepd: error: spank: required plugin spank_pyxis.so: task_init() failed with rc=-1
slurmstepd: error: Failed to invoke spank plugin stack
srun: error: dgx01.i3a.es: task 0: Exited with exit code 1
```

   
To solve this repeating error, the permissions of the files found in ENROOT\_CACHE\_PATH (which was /raid/enroot-cache/ as of December 2023\) should be elevated to allow all users to read. This can be done with admin rights executing:

```
<sudo_user>@dgx01:~$ setfacl -d -m g::r /raid/enroot-cache/group-18000
```

Alternatively, the owner of the cache layer that Pyxis tried to access to can run:

```
<user>@dgx01:~$ chmod g+r /raid/enroot-cache/group-18000/*
```

Although this command does not require sudo access, it does not solve the general problem of ENROOT\_CACHE\_PATH  access rights.  
→ As of December 2023, the first solution was already implemented and, except for a possible DGX-1 reset, there shouldn’t be the necessity to run it again.

### Node in DRAIN state

---

In some circumstances when a job fails the SLURM Node can change to a DRAIN state. In this state, current jobs will be finished, but new jobs won’t be accepted. Running *sinfo* will show the reason of the state change.  
**If this is happening recurrently, contact Dariel from I3A to check the problem, as it could have been caused by a hardware failure.**

To resume the SLURM node to a normal state and start working again, a sudoer should run:

```
<sudo_user>@dgx01:~$ scontrol update nodename=dgx01.i3a.es state=resume
```

