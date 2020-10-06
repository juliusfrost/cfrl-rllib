function iter_to_list(iterator) {
  var l = [];
  while (iterator.hasNext()) {
    l.push(iterator.next());
  }
  return l;
}
