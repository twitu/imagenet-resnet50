import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export class ImagenetStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Create VPC
    const vpc = new ec2.Vpc(this, 'TrainingVPC', {
      maxAzs: 2,
      natGateways: 1,
    });

    // Create S3 bucket
    const bucket = new s3.Bucket(this, 'ImagenetBucket', {
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // Create IAM role for EC2 instances with enhanced permissions
    const role = new iam.Role(this, 'EC2Role', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
    });

    // Add required policies
    role.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore')
    );
    role.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2FullAccess')
    );

    // Grant S3 bucket access
    bucket.grantReadWrite(role);

    // Add custom policy for EBS volume management
    role.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'ec2:AttachVolume',
        'ec2:DetachVolume',
        'ec2:DescribeVolumes',
        'ec2:ModifyVolume',
      ],
      resources: ['*'],
    }));

    // Create security group
    const sg = new ec2.SecurityGroup(this, 'TrainingSecurityGroup', {
      vpc,
      description: 'Security group for training instance',
      allowAllOutbound: true,
    });

    sg.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(22),
      'Allow SSH access'
    );

    // Create EBS volume for dataset
    const dataVolume = new ec2.Volume(this, 'DataVolume', {
      availabilityZone: vpc.availabilityZones[0],
      size: cdk.Size.gibibytes(300),
      volumeType: ec2.EbsDeviceVolumeType.GP3,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Update user data script
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      // System updates and dependencies
      'apt-get update',

      // Get instance ID and volume ID
      'INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)',
      `VOLUME_ID=${dataVolume.volumeId}`,
      'AVAILABILITY_ZONE=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)',

      // Attach volume using AWS CLI
      'aws ec2 attach-volume --volume-id $VOLUME_ID --instance-id $INSTANCE_ID --device /dev/sdf --region ap-south-1',

      // Wait for volume to attach and appear as NVMe device
      'echo "Waiting for volume to attach..."',
      'while [ ! -e /dev/nvme2n1 ]; do sleep 1; done',

      // Debug commands
      'echo "Listing block devices..."',
      'lsblk',
      'ls -l /dev/nvme*',

      // Create mount directory
      'sudo mkdir -p /mnt/training_data',

      // Set permissions
      'sudo chown -R ubuntu:ubuntu /mnt/training_data',
      'sudo chown -R ubuntu:ubuntu /home/ubuntu',

      // Mount volume
      'sudo mount /dev/nvme2n1 /mnt/training_data',
      'df -h /mnt/training_data',

      // Clone and setup repository
      'cd /home/ubuntu',
      'git clone https://github.com/twitu/imagenet-resnet50.git',
      'cd imagenet-resnet50',
      'pip3 install -r requirements.txt',

      // Set environment variables
      `echo "export BUCKET_NAME=${bucket.bucketName}" >> /home/ubuntu/.bashrc`,
      'echo "export DATA_PATH=/mnt/training_data/data" >> /home/ubuntu/.bashrc',
      'echo "export CHECKPOINT_DIR=/mnt/training_data/checkpoints" >> /home/ubuntu/.bashrc',
      'echo "export TRAINING_EPOCHS=100" >> /home/ubuntu/.bashrc',
      'echo "export LEARNING_RATE=0.01" >> /home/ubuntu/.bashrc',
      'echo "export WEIGHT_DECAY=0.05" >> /home/ubuntu/.bashrc',
      'source /home/ubuntu/.bashrc',

      // Start training as ubuntu user with environment variables
      'python3 main.py > training.log 2>&1 &"'
    );

    // Create launch template
    new ec2.LaunchTemplate(this, 'TrainingLaunchTemplate', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.G4DN, ec2.InstanceSize.XLARGE2),
      machineImage: ec2.MachineImage.genericLinux({
        'ap-south-1': 'ami-0979d937ac9f81c9a'
      }),
      blockDevices: [{
        deviceName: '/dev/sda1',
        volume: ec2.BlockDeviceVolume.ebs(80)
      }],
      userData,
      securityGroup: sg,
      role,
      spotOptions: {
        requestType: ec2.SpotRequestType.PERSISTENT,
        interruptionBehavior: ec2.SpotInstanceInterruption.TERMINATE,
        maxPrice: 0.30,
      },
    });

    // Output the bucket name
    new cdk.CfnOutput(this, 'BucketName', {
      value: bucket.bucketName,
    });
  }
}
