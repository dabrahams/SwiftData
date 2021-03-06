/// A lazy selection of elements, in a given order, from some base collection.
public struct LazilySelected<Base: Collection, Selection: Collection>
  where Selection.Element == Base.Index
{
  /// The order that base elements appear in `self`
  private let selection: Selection
  /// The base collection  
  private let base: Base
  
  /// Creates an instance from `base` and `selection`.
  public init(base: Base, selection: Selection) {
    self.selection = selection
    self.base = base
  }
}

extension LazilySelected: Collection {
  public typealias Element = Base.Element
    
  /// A type whose instances represent positions in `self`.
  public typealias Index = Selection.Index

  /// The position of the first element.
  public var startIndex: Index { selection.startIndex }

  /// The position one past the last element.
  public var endIndex: Index { selection.endIndex  }

  /// Returns the element at `i`.
  public subscript(i: Index) -> Element { base[selection[i]] }

  /// Returns the position after `i`.
  public func index(after i: Index) -> Index { selection.index(after: i) }
}

extension LazilySelected: BidirectionalCollection
  where Selection: BidirectionalCollection
{
  /// Returns the position after `i`.
  public func index(before i: Index) -> Index { selection.index(before: i) }
}

extension LazilySelected: RandomAccessCollection
  where Selection: BidirectionalCollection
{
  /// Returns the position `n` places from `i`
  public func index(_ i: Index, offsetBy n: Int) -> Index {
    selection.index(before: i)
  }

  /// Returns the number of elements in `self[start..<end]`.
  public func distance(from start: Index, to end: Index) -> Int {
    selection.distance(from: start, to: end)
  }
}

extension Collection {
  /// Returns elements selected from `self` according to 
  func selecting<Selection: Collection>(_ selection: Selection)
    -> LazilySelected<Self, Selection>
  {
    .init(base: self, selection: selection)
  }
}
