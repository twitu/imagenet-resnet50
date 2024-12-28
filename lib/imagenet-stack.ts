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

    // Create IAM role for EC2 instances
    const role = new iam.Role(this, 'EC2Role', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
    });

    // Add permissions to role
    role.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore')
    );
    
    bucket.grantReadWrite(role);

    // Create security group for both instances
    const sg = new ec2.SecurityGroup(this, 'TrainingSecurityGroup', {
      vpc,
      description: 'Security group for training and upload instances',
      allowAllOutbound: true,
    });

    sg.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(22),
      'Allow SSH access'
    );

    // Create a separate EBS volume for dataset
    const dataVolume = new ec2.Volume(this, 'DataVolume', {
      availabilityZone: vpc.availabilityZones[0], // Use first AZ
      size: cdk.Size.gibibytes(300),
      volumeType: ec2.EbsDeviceVolumeType.GP3,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Create small t3.large instance for data upload
    const trainingInstance = new ec2.Instance(this, 'TrainingInstance', {
      vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PUBLIC,
        availabilityZones: [vpc.availabilityZones[0]], // Must be in same AZ as volume
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.G4DN, ec2.InstanceSize.XLARGE12),
      machineImage: ec2.MachineImage.genericLinux({
        'ap-south-1': 'ami-0979d937ac9f81c9a'
      }),
      role,
      securityGroup: sg,
      keyName: 'imagenet-dataload',
      blockDevices: [{
        deviceName: '/dev/sda1', // Ubuntu uses different device naming
        volume: ec2.BlockDeviceVolume.ebs(80)
      }],
      associatePublicIpAddress: true
    });

    // Attach the volume to the instance
    new ec2.CfnVolumeAttachment(this, 'DataVolumeAttachment', {
      device: '/dev/sdf',
      instanceId: trainingInstance.instanceId,
      volumeId: dataVolume.volumeId,
    });

    // Update user data for Ubuntu
    const uploadUserData = ec2.UserData.forLinux();
    uploadUserData.addCommands(
      'apt-get update',
      'pip3 install boto3',
      `echo "export BUCKET_NAME=${bucket.bucketName}" >> /home/ubuntu/.bashrc`,
      
      // Debug commands to list block devices
      'echo "Listing block devices..."',
      'lsblk',
      'ls -l /dev/nvme*',
      
      // Wait for the EBS volume to be attached
      'while [ ! -e /dev/nvme1n1 ]; do echo waiting for volume to attach; sleep 1; done',
      
      // Debug the volume we found
      'echo "Found volume:"',
      'lsblk /dev/nvme1n1',
      
      // Format the volume if it's not already formatted
      'if [ "$(file -s /dev/nvme1n1)" = "/dev/nvme1n1: data" ]; then',
      '  echo "Formatting new volume..."',
      '  mkfs -t xfs /dev/nvme1n1',
      'else',
      '  echo "Volume is already formatted"',
      'fi',
      
      // Create mount point and mount the volume
      'mkdir -p /data',
      'echo "/dev/nvme1n1 /data xfs defaults,nofail 0 2" >> /etc/fstab',
      'mount /data',
      
      // Verify mount
      'df -h /data',
      
      // Set permissions
      'chown ubuntu:ubuntu /data',
      'chown -R ubuntu:ubuntu /home/ubuntu'
    );

    trainingInstance.addUserData(uploadUserData.render());

    // // Output both instance DNS names and bucket name
    // new cdk.CfnOutput(this, 'TrainingInstanceDNS', {
    //   value: trainingInstance.instancePublicDnsName,
    // });

    new cdk.CfnOutput(this, 'UploadInstanceDNS', {
      value: trainingInstance.instancePublicDnsName,
    });

    new cdk.CfnOutput(this, 'BucketName', {
      value: bucket.bucketName,
    });
  }
}
