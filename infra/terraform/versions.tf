terraform {
  required_version = ">= 1.5"

  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.120"
    }
  }
}

provider "yandex" {
  # Auth via one of:
  #   - token             (yc iam create-token)
  #   - service_account_key_file (path to authorized key JSON)
  # Set whichever you use in terraform.tfvars.
  token                    = var.yc_token != "" ? var.yc_token : null
  service_account_key_file = var.yc_sa_key_file != "" ? var.yc_sa_key_file : null

  cloud_id  = var.yc_cloud_id
  folder_id = var.yc_folder_id
  zone      = var.yc_zone
}
