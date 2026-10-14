# Crée : rôle SSM, groupe de sécurité SANS port entrant, route Internet si elle manque, instance Ubuntu 24.04 m6i.xlarge (100 Go gp3, sans clé SSH),
# amorçage automatique (Docker, git, python3-zmq, clé de déploiement ed25519), puis affiche la clé PUBLIQUE à ajouter à votre dépôt.
# Ne crée aucun secret, #!/usr/bin/env bash
# À COLLER DANS AWS CloudShell (console AWS, icône >_ ; AWS CLI v2 et vos droits de console y sont déjà). Région : Tokyo.
n'ouvre aucun port, ne touche à aucune clé d'exchange.
#
# NON exécuté : écrit sans accès à un compte AWS (bash -n seulement). Lire avant de coller ; en cas d'erreur, la sortie dit laquelle.
set -euo pipefail
export AWS_DEFAULT_REGION=ap-northeast-1
OLD_INSTANCE="${OLD_INSTANCE:-i-003582d21ffd6a197}"   # l'instance morte ; mettre OLD_INSTANCE= (vide) pour ne rien résilier
ROLE=nothierly-ssm

echo "[1/7] ancienne instance"
if [ -n "$OLD_INSTANCE" ]; then
  aws ec2 terminate-instances --instance-ids "$OLD_INSTANCE" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 || echo "(déjà absente ou non résiliable : on continue)"
fi

echo "[2/7] rôle SSM"
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
fi
aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
if ! aws iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
  sleep 15   # propagation IAM
fi

echo "[3/7] réseau"
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
if [ "$VPC" = "None" ]; then
  aws ec2 create-default-vpc >/dev/null
  VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
fi
IGW=$(aws ec2 describe-internet-gateways --filters Name=attachment.vpc-id,Values="$VPC" --query 'InternetGateways[0].InternetGatewayId' --output text)
if [ "$IGW" = "None" ]; then
  IGW=$(aws ec2 create-internet-gateway --query 'InternetGateway.InternetGatewayId' --output text)
  aws ec2 attach-internet-gateway --internet-gateway-id "$IGW" --vpc-id "$VPC"
fi
read -r -a SUBNETS <<<"$(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC" --query 'Subnets[].SubnetId' --output text)"
[ "${#SUBNETS[@]}" -gt 0 ] || { echo "ÉCHEC : aucun sous-réseau dans $VPC"; exit 1; }
for SUBNET in "${SUBNETS[@]}"; do   # chaque sous-réseau doit avoir une route 0.0.0.0/0 vers l'IGW (sinon : pas d'apt, pas de git, pas de SSM)
  RT=$(aws ec2 describe-route-tables --filters Name=association.subnet-id,Values="$SUBNET" --query 'RouteTables[0].RouteTableId' --output text)
  [ "$RT" != "None" ] || RT=$(aws ec2 describe-route-tables --filters Name=vpc-id,Values="$VPC" Name=association.main,Values=true --query 'RouteTables[0].RouteTableId' --output text)
  CURRENT=$(aws ec2 describe-route-tables --route-table-ids "$RT" --query "RouteTables[0].Routes[?DestinationCidrBlock=='0.0.0.0/0'].GatewayId | [0]" --output text)
  if [ "$CURRENT" != "$IGW" ]; then
    echo "  route Internet absente ou cassée sur $RT : ajout 0.0.0.0/0 -> $IGW"
    aws ec2 create-route --route-table-id "$RT" --destination-cidr-block 0.0.0.0/0 --gateway-id "$IGW" >/dev/null 2>&1 \
      || aws ec2 replace-route --route-table-id "$RT" --destination-cidr-block 0.0.0.0/0 --gateway-id "$IGW"
  fi
done
SG=$(aws ec2 create-security-group --group-name "nothierly-sg-$(date +%s)" --description "nothierly : aucun port entrant" --vpc-id "$VPC" --query GroupId --output text)

echo "[4/7] image Ubuntu 24.04"
AMI=$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)
IMAGE_NAME=$(aws ec2 describe-images --image-ids "$AMI" --query 'Images[0].Name' --output text)
case "$IMAGE_NAME" in *ubuntu-noble-24.04-amd64*) echo "  $IMAGE_NAME" ;; *) echo "ÉCHEC : image inattendue ($IMAGE_NAME)"; exit 1 ;; esac

cat > /tmp/nothierly-userdata.sh <<'UD'
#!/bin/bash
set -eux
exec > /var/log/nothierly-bootstrap.log 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ca-certificates curl git python3-zmq
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
usermod -aG docker ubuntu
sudo -u ubuntu bash -c 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && ssh-keygen -q -t ed25519 -f ~/.ssh/nothierly_deploy -N "" -C nothierly-deploy && printf "Host github.com\n  IdentityFile ~/.ssh/nothierly_deploy\n  IdentitiesOnly yes\n" > ~/.ssh/config && chmod 600 ~/.ssh/config && ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null'
touch /var/lib/nothierly-bootstrap.done
UD

echo "[5/7] lancement"
ID=""
for SUBNET in "${SUBNETS[@]}"; do
  if ID=$(aws ec2 run-instances --image-id "$AMI" --instance-type m6i.xlarge --subnet-id "$SUBNET" --security-group-ids "$SG" \
        --associate-public-ip-address --iam-instance-profile Name="$ROLE" \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled --user-data file:///tmp/nothierly-userdata.sh \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=nothierly}]' --query 'Instances[0].InstanceId' --output text 2>/tmp/run.err); then
    break
  fi
  echo "  $SUBNET refusé : $(tail -1 /tmp/run.err)"; ID=""
done
[ -n "$ID" ] || { echo "ÉCHEC : aucune zone n'a accepté m6i.xlarge"; exit 1; }
echo "  instance $ID"

echo "[6/7] attente de l'instance et de l'agent SSM (3 à 6 minutes)"
aws ec2 wait instance-running --instance-ids "$ID"
for _ in $(seq 1 60); do
  PING=$(aws ssm describe-instance-information --filters Key=InstanceIds,Values="$ID" --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
  [ "$PING" = "Online" ] && break
  sleep 10
done
[ "$PING" = "Online" ] || { echo "SSM n'est pas en ligne : vérifier la route Internet et le rôle $ROLE (EC2 > instance > Sécurité)."; exit 1; }

echo "[7/7] clé de déploiement (PUBLIQUE) et empreinte de github.com"
CMD=$(aws ssm send-command --instance-ids "$ID" --document-name AWS-RunShellScript --timeout-seconds 900 \
  --parameters 'commands=["for i in $(seq 1 180); do [ -f /var/lib/nothierly-bootstrap.done ] && break; sleep 5; done","test -f /var/lib/nothierly-bootstrap.done || { echo BOOTSTRAP_INCOMPLET; tail -20 /var/log/nothierly-bootstrap.log; exit 1; }","cat /home/ubuntu/.ssh/nothierly_deploy.pub","ssh-keygen -lf /home/ubuntu/.ssh/known_hosts","docker --version"]' \
  --query Command.CommandId --output text)
sleep 20
for _ in $(seq 1 60); do
  STATUS=$(aws ssm get-command-invocation --command-id "$CMD" --instance-id "$ID" --query Status --output text 2>/dev/null || echo Pending)
  case "$STATUS" in Success|Failed|Cancelled|TimedOut) break ;; esac
  sleep 10
done
aws ssm get-command-invocation --command-id "$CMD" --instance-id "$ID" --query StandardOutputContent --output text
echo
echo "INSTANCE : $ID  (terminal : EC2 > Instances > cocher > Se connecter > Gestionnaire de session)"
echo "SUITE : ajouter la clé publique ci-dessus aux Deploy keys du dépôt (lecture seule), puis dans le terminal SSM : sudo su - ubuntu"
