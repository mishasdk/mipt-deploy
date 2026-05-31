# Terraform — Yandex Cloud VM

Поднимает Ubuntu 22.04 VM в Yandex Cloud (2 vCPU @ core_fraction 20%, 8 ГБ RAM)
и устанавливает Docker + docker compose plugin через cloud-init.

## Создание инфры

```bash
terraform init
terraform plan
terraform apply
```

## Удаление инфры

```bash
terraform destroy
```

Удаляет VM, диск, подсеть и сеть.
