"""Unit tests for data primitives"""

import numpy as np
from artifacta.primitives import Distribution, Graph, Hierarchy, Matrix, Series, Table


class TestSeries:
    """Test Series primitive"""

    def test_series_creation(self):
        """Should create series with required fields"""
        s = Series(index="epoch", fields={"loss": [1.0, 0.5, 0.2], "accuracy": [0.6, 0.8, 0.9]})

        assert s.index == "epoch"
        assert s.fields == {"loss": [1.0, 0.5, 0.2], "accuracy": [0.6, 0.8, 0.9]}

    def test_series_to_dict(self):
        """Should convert to dict correctly"""
        s = Series(index="step", fields={"metric": [1, 2, 3]}, index_values=[0, 10, 20])
        d = s.to_dict()

        assert d["index"] == "step"
        assert d["fields"] == {"metric": [1, 2, 3]}
        assert d["index_values"] == [0, 10, 20]

    def test_series_with_numpy(self):
        """Should handle numpy arrays"""
        s = Series(
            index="time",
            fields={"values": np.array([1.0, 2.0, 3.0])},
            index_values=np.array([0, 1, 2]),
        )
        d = s.to_dict()

        # Should convert numpy to lists
        assert isinstance(d["fields"]["values"], list)
        assert isinstance(d["index_values"], list)
        assert d["fields"]["values"] == [1.0, 2.0, 3.0]

    def test_series_with_metadata(self):
        """Should include metadata"""
        s = Series(
            index="epoch",
            fields={"loss": [1.0]},
            metadata={"model": "resnet50", "dataset": "imagenet"},
        )
        d = s.to_dict()

        assert d["metadata"] == {"model": "resnet50", "dataset": "imagenet"}


class TestDistribution:
    """Test Distribution primitive"""

    def test_distribution_creation(self):
        """Should create distribution with values"""
        dist = Distribution(values=[0.1, 0.2, 0.3, 0.4])

        assert dist.values == [0.1, 0.2, 0.3, 0.4]
        assert dist.groups is None

    def test_distribution_with_groups(self):
        """Should support grouped distributions"""
        dist = Distribution(values=[0.1, 0.2, 0.15, 0.25], groups=["A", "B", "A", "B"])

        assert dist.groups == ["A", "B", "A", "B"]

    def test_distribution_to_dict(self):
        """Should convert to dict correctly"""
        dist = Distribution(values=[1.0, 2.0, 3.0], groups=["x", "y", "z"], metadata={"bins": 10})
        d = dist.to_dict()

        assert d["values"] == [1.0, 2.0, 3.0]
        assert d["groups"] == ["x", "y", "z"]
        assert d["metadata"] == {"bins": 10}

    def test_distribution_with_numpy(self):
        """Should handle numpy arrays"""
        dist = Distribution(values=np.array([1.0, 2.0, 3.0]), groups=np.array(["a", "b", "c"]))
        d = dist.to_dict()

        assert isinstance(d["values"], list)
        assert isinstance(d["groups"], list)


class TestMatrix:
    """Test Matrix primitive"""

    def test_matrix_creation(self):
        """Should create matrix with rows, cols, values"""
        m = Matrix(rows=["A", "B"], cols=["X", "Y"], values=[[1, 2], [3, 4]])

        assert m.rows == ["A", "B"]
        assert m.cols == ["X", "Y"]
        assert m.values == [[1, 2], [3, 4]]

    def test_matrix_to_dict(self):
        """Should convert to dict correctly"""
        m = Matrix(
            rows=["cat", "dog"],
            cols=["cat", "dog"],
            values=[[10, 2], [1, 12]],
            metadata={"accuracy": 0.92},
        )
        d = m.to_dict()

        assert d["rows"] == ["cat", "dog"]
        assert d["cols"] == ["cat", "dog"]
        assert d["values"] == [[10, 2], [1, 12]]
        assert d["metadata"] == {"accuracy": 0.92}

    def test_matrix_with_numpy(self):
        """Should handle numpy arrays"""
        m = Matrix(rows=["A"], cols=["B"], values=np.array([[5]]))
        d = m.to_dict()

        assert isinstance(d["values"], list)


class TestGraph:
    """Test Graph primitive"""

    def test_graph_creation(self):
        """Should create graph with nodes and edges"""
        g = Graph(
            nodes=[{"id": "a", "label": "Node A"}, {"id": "b", "label": "Node B"}],
            edges=[{"source": "a", "target": "b"}],
        )

        assert len(g.nodes) == 2
        assert len(g.edges) == 1
        assert g.nodes[0]["id"] == "a"
        assert g.edges[0]["source"] == "a"

    def test_graph_to_dict(self):
        """Should convert to dict correctly"""
        g = Graph(
            nodes=[{"id": "1"}], edges=[{"source": "1", "target": "2"}], metadata={"directed": True}
        )
        d = g.to_dict()

        assert d["nodes"] == [{"id": "1"}]
        assert d["edges"] == [{"source": "1", "target": "2"}]
        assert d["metadata"] == {"directed": True}


class TestTable:
    """Test Table primitive"""

    def test_table_creation(self):
        """Should create table with columns and data"""
        t = Table(
            columns=[
                {"name": "Name", "type": "string"},
                {"name": "Age", "type": "number"},
                {"name": "Score", "type": "number"},
            ],
            data=[["Alice", 25, 95.5], ["Bob", 30, 88.0]],
        )

        assert len(t.columns) == 3
        assert len(t.data) == 2

    def test_table_to_dict(self):
        """Should convert to dict correctly"""
        t = Table(
            columns=[{"name": "X", "type": "int"}, {"name": "Y", "type": "int"}],
            data=[[1, 2], [3, 4]],
            metadata={"source": "experiment"},
        )
        d = t.to_dict()

        assert d["columns"] == [{"name": "X", "type": "int"}, {"name": "Y", "type": "int"}]
        assert d["data"] == [[1, 2], [3, 4]]
        assert d["metadata"] == {"source": "experiment"}

    def test_table_empty(self):
        """Should handle empty table"""
        t = Table(
            columns=[{"name": "A", "type": "string"}, {"name": "B", "type": "string"}], data=[]
        )
        d = t.to_dict()

        assert len(d["columns"]) == 2
        assert d["data"] == []


class TestHierarchy:
    """Test Hierarchy primitive"""

    def test_hierarchy_creation(self):
        """Should create hierarchy with nodes (parent-child encoded in nodes)"""
        h = Hierarchy(
            nodes=[
                {"id": "root", "label": "Root"},
                {"id": "child1", "label": "Child 1", "parent": "root"},
                {"id": "child2", "label": "Child 2", "parent": "root"},
            ]
        )

        assert len(h.nodes) == 3
        assert h.nodes[0]["id"] == "root"
        assert h.nodes[1]["parent"] == "root"

    def test_hierarchy_to_dict(self):
        """Should convert to dict correctly"""
        h = Hierarchy(nodes=[{"id": "1", "label": "Node 1"}], metadata={"type": "tree"})
        d = h.to_dict()

        assert d["nodes"] == [{"id": "1", "label": "Node 1"}]
        assert d["metadata"] == {"type": "tree"}


class TestPrimitiveIntegration:
    """Test integration between primitives"""

    def test_all_primitives_have_to_dict(self):
        """All primitives should have to_dict method"""
        primitives = [
            Series(index="x", fields={"y": [1]}),
            Distribution(values=[1, 2, 3]),
            Matrix(rows=["A"], cols=["B"], values=[[1]]),
            Graph(nodes=[{"id": "1"}], edges=[]),
            Table(columns=[{"name": "X", "type": "int"}], data=[[1]]),
            Hierarchy(nodes=[{"id": "1"}]),
        ]

        for p in primitives:
            assert hasattr(p, "to_dict")
            d = p.to_dict()
            assert isinstance(d, dict)

    def test_all_primitives_support_metadata(self):
        """All primitives should support metadata field"""
        primitives = [
            Series(index="x", fields={"y": [1]}, metadata={"test": True}),
            Distribution(values=[1], metadata={"test": True}),
            Matrix(rows=["A"], cols=["B"], values=[[1]], metadata={"test": True}),
            Graph(nodes=[], edges=[], metadata={"test": True}),
            Table(columns=[{"name": "X", "type": "int"}], data=[], metadata={"test": True}),
            Hierarchy(nodes=[], metadata={"test": True}),
        ]

        for p in primitives:
            d = p.to_dict()
            assert "metadata" in d
            assert d["metadata"] == {"test": True}
