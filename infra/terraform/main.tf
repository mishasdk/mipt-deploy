locals {
  create_network = var.existing_subnet_id == ""
  subnet_id      = local.create_network ? yandex_vpc_subnet.this[0].id : var.existing_subnet_id
  zone           = local.create_network ? var.yc_zone : data.yandex_vpc_subnet.existing[0].zone
}

data "yandex_vpc_subnet" "existing" {
  count     = local.create_network ? 0 : 1
  subnet_id = var.existing_subnet_id
}

data "yandex_compute_image" "ubuntu" {
  family = var.image_family
}

resource "yandex_vpc_network" "this" {
  count = local.create_network ? 1 : 0
  name  = "${var.vm_name}-net"
}

resource "yandex_vpc_subnet" "this" {
  count          = local.create_network ? 1 : 0
  name           = "${var.vm_name}-subnet"
  zone           = var.yc_zone
  network_id     = yandex_vpc_network.this[0].id
  v4_cidr_blocks = ["10.10.0.0/24"]
}

resource "yandex_compute_instance" "this" {
  name        = var.vm_name
  platform_id = "standard-v3"
  zone        = local.zone

  resources {
    cores         = var.vm_cores
    core_fraction = var.vm_core_fraction
    memory        = var.vm_memory_gb
  }

  scheduling_policy {
    preemptible = var.preemptible
  }

  boot_disk {
    initialize_params {
      image_id = data.yandex_compute_image.ubuntu.id
      size     = var.vm_disk_gb
    }
  }

  network_interface {
    subnet_id = local.subnet_id
    nat       = true
  }

  metadata = {
    user-data = templatefile("${path.module}/cloud-init.yaml", {
      ssh_user       = var.ssh_user
      ssh_public_key = trimspace(file(pathexpand(var.ssh_public_key_path)))
    })
  }
}
