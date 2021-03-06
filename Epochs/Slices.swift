/// A collection of the longest non-overlapping contiguous slices of some `Base`
/// collection, starting with its first element, and having some fixed maximum
/// length.
///
/// The elements of this collection, except for the last, all have a `count` of
/// `maxLength`.  The last one's `count` is `base.count % maxLength.`
public struct Slices<Base: Collection> {
  /// The collection from which slices will be drawn.
  private let base: Base
  
  /// The maximum length of the slices
  private let batchSize: Int

  public init(_ base: Base, batchSize: Int) {
    self.base = base
    self.batchSize = batchSize
  }
}

extension Slices : Collection {
  /// A position in `Slices`.
  public struct Index : Comparable {
    /// The range of base indices covered by the element at this position.
    var focus: Range<Base.Index>

    /// Returns true if `l` precedes `r` in the collection.
    public static func < (l: Index, r: Index) -> Bool {
      l.focus.lowerBound < r.focus.lowerBound
    }
  }

  /// Returns the element at `i`.
  public subscript(i: Index) -> Base.SubSequence { base[i.focus] }

  /// Returns the base index that marks the end of the element of `self` that
  /// begins at `i` in the `base`, or `base.endIndex` if `i == base.endIndex`.
  private func sliceBoundary(after i: Base.Index) -> Base.Index {
    base.index(i, offsetBy: batchSize, limitedBy: base.endIndex)
      ?? base.endIndex
  }
  
  /// Returns the index after `i`.
  public func index(after i: Index) -> Index {
    Index(focus: i.focus.upperBound..<sliceBoundary(after: i.focus.upperBound))
  }

  /// Returns the first position ini `self`.
  public var startIndex: Index {
    Index(focus: base.startIndex..<sliceBoundary(after: base.startIndex))
  }

  /// Returns the position one past the last element of `self`.
  public var endIndex: Index {
    Index(focus: base.endIndex..<base.endIndex)
  }
}

extension Collection {
  /// Returns the longest non-overlapping slices of `self`, starting with its
  /// first element, and having a maximum length of `batchSize`.
  public func inBatches(of batchSize: Int) -> Slices<Self> {
    Slices(self, batchSize: batchSize)
  }
}

// FIXME: Batches is now just
//
//   Slices(batchSamples, elementMaxLength: batchSize).lazy.map(transform)
//
// so we should refactor :-)
