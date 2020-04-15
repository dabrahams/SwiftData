import TensorFlow
import Epochs
import XCTest

import Batcher1

class Batcher1Tests : XCTestCase {
  // Base use
  func test0(){
    // Some raw items (for instance filenames)
    let rawItems = 0..<512
    // A heavy-compute function lazily mapped on it (for instance, opening the images)
    let dataSet = rawItems.lazy.map { _ in Tensor<Float>(randomNormal: [224, 224, 3]) }
    // A `Batcher` defined on this:
    var batcher = Batcher(on: dataSet, batchSize: 64)
    // Iteration over it (for instance, on epoch of the training loop) 
    batcher.reorder(shuffled: true)
    for batch in batcher {
      print(batch.shape)
    }
  }

  // Use with padding
  // Let's create an array of things of various lengths (for instance texts)
  let dataSet: [Tensor<Int32>] = { 
    var dataSet: [Tensor<Int32>] = []
    for _ in 0..<512 {
      dataSet.append(Tensor<Int32>(
        randomUniform: [Int.random(in: 1...200)], 
        lowerBound: Tensor<Int32>(0), 
        upperBound: Tensor<Int32>(100)
      ))
    }
    return dataSet
  }()

  // We need to pad those tensors to make them all the same length.
  // We could do this in one lazy transform applied beforehand and pad everything
  // to the same length, but it's not memory-efficient: some batches might need less
  // padding. So we need to add the padding after having selected the samples we
  // are trying to batch.
  func padTensors(tensors: [Tensor<Int32>]) -> [Tensor<Int32>] {
    let maxLength = tensors.map{ $0.shape[0] }.max()!
    return tensors.map { (t: Tensor<Int32>) -> Tensor<Int32> in 
      let remaining = Tensor<Int32>(zeros: [maxLength - t.shape[0]])
      return Tensor<Int32>(concatenating: [t, remaining])
    }
  } 
    
  func test1() {
    var batcher = Batcher(on: dataSet, batchSize: 64, padSamples: padTensors)
    batcher.reorder()
    for b in batcher {
      print(b.shape)
    }
  }

  // Use with a sampler
  // In our previous example, another way to be memory efficient is to batch
  // samples of roughly the same lengths.
  func sortSamples(on dataset: [Tensor<Int32>]) -> [Int] {
    return Array(0..<dataset.count).sorted { dataset[$0].shape[0] > dataset[$1].shape[0] }
  }

  func test2() {
    // Use with a sampler
    // In our previous example, another way to be memory efficient is to batch
    // samples of roughly the same lengths.
    var batcher = Batcher(on: dataSet, batchSize: 64, sampleIndices: sortSamples, padSamples: padTensors)
    batcher.reorder()
    for b in batcher {
      print(b.shape)
    }
  }

  // Sometimes the shuffle method needs to be applied on the dataset itself, 
  // like for language modeling. Here is a base version of that to test
  // the API allows it.
  struct DataSet: RandomAccessCollection {
    typealias Index = Int
    typealias Element = Tensor<Int32>
    
    let numbers: [[Int]]
    let sequenceLength: Int
    // The texts all concatenated together
    private var stream: [Int]
    
    var startIndex: Int { return 0 }
    var endIndex: Int { return stream.count / sequenceLength }
    func index(after i: Int) -> Int { i+1 }
    
    init(numbers: [[Int]], sequenceLength: Int) {
      self.numbers = numbers
      self.sequenceLength = sequenceLength
      stream = numbers.reduce([], +)
    }
    
    subscript(index: Int) -> Tensor<Int32> {
      get { 
        let i = index * sequenceLength
        return Tensor<Int32>(stream[i..<i+sequenceLength].map { Int32($0)} )
      }
    }
    
    mutating func shuffle() {
      stream = numbers.shuffled().reduce([], +)
    }
  }

  //Let's create such a DataSet
  let dataset: DataSet = {
    let numbers: [[Int]] = [[1,2,3,4,5], [6,7,8], [9,10,11,12,13,14,15], [16,17,18]]
    return DataSet(numbers: numbers, sequenceLength: 3)
  }()

  // This is the shuffle function we will use: it always returns the default indices
  // and shuffles the dataset if needed
  func internalShuffle(on dataset: inout DataSet, indices: [Int]) -> [Int] {
    dataset.shuffle()
    return indices
  }

  //Now let's look at what it gives us:
  func test3() {
    var batcher = Batcher(on: dataset, batchSize: 3, shuffleIndices: internalShuffle)
    batcher.reorder()
    for b in batcher {
      print(b)
    }
      
    batcher.reorder(shuffled: true)
    for b in batcher {
      print(b)
    }
  }
}

extension Batcher1Tests {
  static var allTests = [
    ("test0", test0),
    ("test1", test1),
    ("test2", test2),
    ("test3", test3),
  ]
}

final class EpochsTests : XCTestCase {
  // Some raw items (for instance filenames)
  let rawItems: [Int] = Array(0..<512)
  
  func testLazyShuffle() {
    // A lazy dataset and an array that keeps track of whether the elements were
    // accessed or not.
    var accessed = rawItems.map { _ in false }
    let dataset = rawItems.lazy.map { (x: Int) -> Tensor<Float> in
      accessed[x] = true
      return Tensor<Float>(randomNormal: [224, 224, 3])
    }
    
    // Using `.shuffled()` access all elements
    let _ = dataset.shuffled()
    XCTAssert(accessed.allSatisfy { $0 })
      
    // Using `.innerShuffled()` on a `ReindexedColletion` does not access elements
    accessed = rawItems.map { _ in false }
    let _ = ReindexedCollection(dataset).innerShuffled()
    XCTAssertFalse(accessed.contains(true))
  }
  
  func testBaseUse() {
   // A heavy-compute function lazily mapped on the raw items (for instance, opening the images)
    let dataSet = rawItems.lazy.map { _ in Tensor<Float>(randomNormal: [224, 224, 3]) }

    // A `Batcher` defined on this:
    let batches = Batches(of: 64, from: dataSet, \.collated)
    // Iteration over it (for instance, on epoch of the training loop) 
    XCTAssert(batches.allSatisfy() { 
      $0.shape == TensorShape([64, 224, 224, 3]) }
    )
  }
    
  // Test with shuffle
  func test1() {
    // We need to actually go back to the raw collection to shuffle:
    let dataSet = rawItems.shuffled().lazy.map { _ -> Tensor<Float> in
      return Tensor<Float>(randomNormal: [224, 224, 3])
    }
    let batches = Batches(of: 64, from: dataSet, \.collated)
    print(batches.map(\.shape))
  }
    
  func test2() {
    // ReindexCollection does that for us
    let dataSet = rawItems.lazy.map { _ -> Tensor<Float> in
      return Tensor<Float>(randomNormal: [224, 224, 3])
    }
    let batches = Batches(of: 64, from: ReindexedCollection(dataSet).innerShuffled(), \.collated)
    print(batches.map(\.shape))
  }
    
  // Use with padding
  // Let's create an array of things of various lengths (for instance texts)
  let dataSet: [Tensor<Int32>] = { 
    var dataSet: [Tensor<Int32>] = []
    for _ in 0..<512 {
      dataSet.append(Tensor<Int32>(
                       randomUniform: [Int.random(in: 150...200)], 
                       lowerBound: Tensor<Int32>(0), 
                       upperBound: Tensor<Int32>(100)
                    ))
    }
    return dataSet
  }()
    
  func paddingTest(padValue: Int32, padFirst: Bool) {
    let batches = Batches(of: 64, from: dataSet) { $0.paddedAndCollated(with: padValue) }
    for (i, b) in batches.enumerated() {
      let shapes = dataSet[(i * 64)..<((i + 1) * 64)].map { Int($0.shape[0]) }
      let expectedShape = shapes.reduce(0) { max($0, $1) }
      XCTAssertEqual(Int(b.shape[1]), expectedShape)
        
      for k in 0..<64 {
        let currentShape = dataSet[i * 64 + k].shape[0]
        XCTAssertEqual(
          b[k, currentShape..<expectedShape], 
          Tensor<Int32>(repeating: padValue, shape: [expectedShape - currentShape]))         
      }
    }
  }

  func testAllPadding() {
    paddingTest(padValue: 0, padFirst: false)
    paddingTest(padValue: 42, padFirst: false)
    paddingTest(padValue: 0, padFirst: true)
    paddingTest(padValue: -1, padFirst: true)
  }

  // Use with a sampler
  // In our previous example, another way to be memory efficient is to batch
  // samples of roughly the same lengths.
  func sortSamples(on dataset: inout [Tensor<Int32>], shuffled: Bool) -> [Int] {
    // Just giving a quick example, but the function should shuffle a bit the sorted thing
    // if shuffle=true
    return Array(0..<dataset.count).sorted { dataset[$0].shape[0] > dataset[$1].shape[0] }
  }

  func test4() {
    // Use with a sampler
    // In our previous example, another way to be memory efficient is to batch
   // samples of roughly the same lengths.
    let sortedDataset = dataSet.sorted { $0.shape[0] > $1.shape[0] }
      
    let batches = Batches(of: 64, from: sortedDataset) { $0.paddedAndCollated(with: 0) }
    print(batches.map(\.shape))
  }
    
  func test5() {
    // When using a `batchSize` we get a bit of shuffle:
    // This can all be applied on a lazy collection without breaking the lasziness as long as the sort function does not access the dataset
    let sortedDataset = ReindexedCollection(dataSet).innerShuffled().sortedInBatches(of: 256) { 
      dataSet[$0].shape[0] > dataSet[$1].shape[0] 
    }

    let batches = Batches(of: 64, from: sortedDataset) { $0.paddedAndCollated(with: 0) }
    print(batches.map(\.shape))
  }
    
  struct LanguageModelDataset<Texts: RandomAccessCollection>: Collection where Texts.Element == [Int] {
    /// The underlying collection of texts
    public var texts: Texts
    /// The length of the samples returned when indexing
    private let sequenceLength: Int
    // The texts all concatenated together
    private var stream: [Int]
    
    init(texts: Texts, sequenceLength: Int) {
      self.texts = texts
      self.sequenceLength = sequenceLength
      stream = texts.reduce([], +)
    }
    
    public typealias Index = Int
    public typealias Element = Tensor<Int32>
    
    public var startIndex: Int { return 0 }
    public var endIndex: Int { return stream.count / sequenceLength }
    public func index(after i: Int) -> Int { i+1 }
    
    public subscript(index: Int) -> Tensor<Int32> {
      get { 
        let i = index * sequenceLength
        return Tensor<Int32>(stream[i..<i+sequenceLength].map { Int32($0)} )
      }
    }
  }
    
  //Now let's look at what it gives us:
  func test6() {
    let numbers: [[Int]] = [[1,2,3,4,5], [6,7,8], [9,10,11,12,13,14,15], [16,17,18]]
    let languageDataset = LanguageModelDataset(texts: numbers, sequenceLength: 3)
    let batches = Batches(of: 3, from: languageDataset, \.collated)
    print(batches.map(\.shape))
  }

  func test7() {
    // To shuffle it we need to go back to the inner numbers
    let numbers: [[Int]] = [[1,2,3,4,5], [6,7,8], [9,10,11,12,13,14,15], [16,17,18]]
    let languageDataset = LanguageModelDataset(texts: numbers.shuffled(), sequenceLength: 3)
    let batches = Batches(of: 3, from: languageDataset, \.collated)
    print(batches.map(\.shape))
  }
}


extension EpochsTests {
  static var allTests = [
    ("testLazyShuffle", testLazyShuffle),
    ("testBaseUse", testBaseUse),
    ("test1", test1),
    ("test2", test2),
    ("testAllPadding", testAllPadding),
    ("test4", test4),
    ("test5", test5),
    ("test6", test6),
    ("test7", test7),
  ]
}
