class PointToggle extends React.Component {
    render() {
        const handleClick = () => {
          this.props.handlePointToggle(this.props.num);
        }

        return (
            <div className="m-0 p-0 w-100" style={{outlineStyle: "solid", outlineWidth: 1}}>
                <button type="button"
                        className="btn btn-sm btn-primary row m-0 w-100"
                        onClick={handleClick} enabled>Point {this.props.num + 1}
                </button>
            </div>
        )
    }
}