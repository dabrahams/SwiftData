// swift-tools-version:5.2

import PackageDescription

let package = Package(
  name: "SwiftData",
  platforms: [ .macOS(.v10_13) ],
  products: [
    .library(name: "Batcher", targets: ["Batcher"]),
    .library(name: "Batcher1", targets: ["Batcher1"]),
    .library(name: "Batcher2", targets: ["Batcher2"]),
    .library(name: "Epochs", targets: ["Epochs"])
  ],
  targets: [
    .target(name: "Batcher", path: "Batcher"),
//    .testTarget(name: "BatcherTest", dependencies: ["Batcher"]),
    .target(name: "Batcher1", path: "Batcher1"),
//    .testTarget(name: "Batcher1Test", dependencies: ["Batcher1"]),
    .target(name: "Batcher2", path: "Batcher2"),
//    .testTarget(name: "Batcher2Test", dependencies: ["Batcher2"]),
    .target(name: "Epochs", path: "Epochs"),
    .testTarget(name: "EpochsTest", dependencies: ["Epochs", "Batcher1"]),
  ])
